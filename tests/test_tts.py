import io
import json
import wave
from types import SimpleNamespace as NS

import httpx
import pytest

import config
import tts
from conftest import client  # noqa: F401


def test_voice_map_filters_and_validates(tmp_path):
    f = tmp_path / "voices.json"
    f.write_text(json.dumps({"en": {"engine": "piper", "model": "m"}, "xx": {"engine": "piper"}}))
    assert tts.load_voice_map(f) == {"en": {"engine": "piper", "model": "m"}}
    f.write_text(json.dumps({"ar-SA": {"engine": "nope"}}))
    with pytest.raises(ValueError):
        tts.load_voice_map(f)


def test_voice_map_default_without_file(tmp_path):
    assert tts.load_voice_map(tmp_path / "missing.json") == {"en": {"engine": "piper", "model": config.PIPER_VOICE}}


def test_speak_uses_dialect_engine(monkeypatch):
    spoken = []
    monkeypatch.setattr(tts, "get_engine", lambda d: NS(synthesize=lambda t: spoken.append((d, t)) or b"RIFF"))
    r = client.post("/api/speak", json={"text": "هلا", "dialect": "ar-SA"})
    assert r.status_code == 200 and r.content == b"RIFF" and spoken == [("ar-SA", "هلا")]


def test_speak_missing_dialect_voice(monkeypatch):
    def none(d):
        raise KeyError(d)

    monkeypatch.setattr(tts, "get_engine", none)
    r = client.post("/api/speak", json={"text": "هلا", "dialect": "ar-MA"})
    assert r.status_code == 503 and r.json() == {"detail": "No voice for this dialect."}


def test_speak_rejects_unknown_dialect():
    assert client.post("/api/speak", json={"text": "hi", "dialect": "../x"}).status_code == 422


def test_speak_arabic_length_bound():
    assert client.post("/api/speak", json={"text": "س" * 1001, "dialect": "ar-SA"}).status_code == 422


def test_health_lists_voices(monkeypatch):
    monkeypatch.setattr(tts, "available_dialects", lambda: ["en", "ar-SA"])
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk")
    assert client.get("/api/health").json() == {
        "deepseek_key": True, "voices": {"en": True, "ar-MA": False, "ar-EG": False, "ar-SA": True}}


def fake_elevenlabs(monkeypatch, pcm=b"\x01\x00\x02\x00" * 100, content_type="audio/pcm"):
    seen = {}

    def post(url, **kw):
        seen["url"] = url
        seen.update(kw)
        return httpx.Response(200, content=pcm, headers={"content-type": content_type},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "xi-test")
    return seen


def test_elevenlabs_asks_for_pcm_in_the_query_string(monkeypatch):
    # live bug: output_format in the JSON body is ignored, ElevenLabs answered with MP3, and the MP3
    # bytes were wrapped as PCM, which plays as short loud noise
    pcm = b"\x01\x00\x02\x00" * 100
    seen = fake_elevenlabs(monkeypatch, pcm)
    wav = tts._elevenlabs_engine({"voice_id": "VOICE1"}).synthesize("hello")
    assert seen["url"] == "https://api.elevenlabs.io/v1/text-to-speech/VOICE1"
    assert seen["params"] == {"output_format": "pcm_24000"}
    assert seen["json"] == {"text": "hello", "model_id": "eleven_multilingual_v2"}
    assert seen["headers"]["xi-api-key"] == "xi-test"
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (24000, 1, 2)
        assert w.readframes(w.getnframes()) == pcm


def test_elevenlabs_model_comes_from_the_voice_spec(monkeypatch):
    seen = fake_elevenlabs(monkeypatch)
    tts._elevenlabs_engine({"voice_id": "V", "model_id": "eleven_flash_v2_5"}).synthesize("hi")
    assert seen["json"]["model_id"] == "eleven_flash_v2_5"


@pytest.mark.parametrize("content_type", ["audio/mpeg", "application/json"])
def test_elevenlabs_rejects_a_reply_that_is_not_pcm(monkeypatch, content_type):
    # an MP3 or a JSON error body must fail loudly instead of being wrapped and played as noise
    fake_elevenlabs(monkeypatch, pcm=b"ID3\x04" + b"\x00" * 50, content_type=content_type)
    with pytest.raises(ValueError):
        tts._elevenlabs_engine({"voice_id": "V"}).synthesize("hi")


def test_speak_voice_load_failure(monkeypatch):
    def boom(d):
        raise RuntimeError("corrupt")

    monkeypatch.setattr(tts, "get_engine", boom)
    r = client.post("/api/speak", json={"text": "Hi", "dialect": "en"})
    assert r.status_code == 503 and r.json() == {"detail": "Voice unavailable."}


def test_speak_synthesis_failure(monkeypatch):
    monkeypatch.setattr(tts, "get_engine", lambda d: None)

    def boom(text, dialect):
        raise RuntimeError("no audio")

    monkeypatch.setattr(tts, "synthesize", boom)
    r = client.post("/api/speak", json={"text": "Hi", "dialect": "en"})
    assert r.status_code == 500 and r.json() == {"detail": "Speech synthesis failed."}
