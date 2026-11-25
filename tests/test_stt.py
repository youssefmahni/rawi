import sys
import types
from types import SimpleNamespace as NS

import httpx
import pytest

import config
import stt
from conftest import client  # noqa: F401


@pytest.fixture(autouse=True)
def whisper_engine_by_default(monkeypatch):
    """The Whisper tests below exercise the local engine; the ElevenLabs tests switch it explicitly."""
    monkeypatch.setattr(config, "STT_ENGINE", "whisper")


def info(lang, probs):
    return NS(language=lang, all_language_probs=probs)


class Seg:
    def __init__(self, text):
        self.text = text


def fake_whisper(monkeypatch, runs):
    """runs: list of (segments, info) returned by successive transcribe() calls; records kwargs."""
    calls = []

    class M:
        def transcribe(self, audio, **kw):
            calls.append(kw)
            segs, inf = runs[len(calls) - 1]
            return iter([Seg(s) for s in segs]), inf

    monkeypatch.setattr(stt, "get_whisper", lambda: M())
    return calls


def test_detects_arabic_in_one_pass(monkeypatch):
    calls = fake_whisper(monkeypatch, [([" وش", " أخبارك؟"], info("ar", [("ar", .9)]))])
    assert stt.transcribe_audio(b"x") == stt.Transcript("وش أخبارك؟", "ar")
    assert calls == [{"language": None, "vad_filter": True, "beam_size": 1}]


def test_non_en_ar_detection_retries_with_best_of_en_ar(monkeypatch):
    calls = fake_whisper(monkeypatch, [(["bonjour"], info("fr", [("fr", .5), ("ar", .3), ("en", .1)])),
                                       ([" كيداير"], info("ar", []))])
    assert stt.transcribe_audio(b"x") == stt.Transcript("كيداير", "ar")
    assert calls[1]["language"] == "ar"


def test_tie_goes_to_arabic(monkeypatch):
    calls = fake_whisper(monkeypatch, [(["x"], info("fa", [("en", .2), ("ar", .2)])), ([" y"], info("ar", []))])
    stt.transcribe_audio(b"x")
    assert calls[1]["language"] == "ar"


def test_english(monkeypatch):
    fake_whisper(monkeypatch, [([" Hi there."], info("en", [("en", .99)]))])
    assert stt.transcribe_audio(b"x") == stt.Transcript("Hi there.", "en")


def test_transcribe_route_returns_language(monkeypatch):
    monkeypatch.setattr(stt, "transcribe_audio", lambda d: stt.Transcript("هلا", "ar"))
    monkeypatch.setattr(stt, "get_transcriber", lambda: None)
    assert client.post("/api/transcribe", content=b"a").json() == {"text": "هلا", "language": "ar"}


def test_empty_body_has_language():
    assert client.post("/api/transcribe", content=b"").json() == {"text": "", "language": "en"}


def test_model_settings_cuda(monkeypatch):
    seen = {}

    class WM:
        def __init__(self, name, **kw):
            seen.update(kw, name=name)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WM))
    monkeypatch.setattr(config, "STT_DEVICE", "cuda")
    monkeypatch.setattr(config, "STT_COMPUTE", "")
    stt.get_whisper.cache_clear()
    try:
        stt.get_whisper()
    finally:
        stt.get_whisper.cache_clear()
    assert seen["device"] == "cuda" and seen["compute_type"] == "float16" and seen["name"] == config.STT_MODEL


# ---- ElevenLabs Scribe ----
def fake_scribe(monkeypatch, status=200, body=None, exc=None, key="xi-test"):
    seen = {}

    def post(url, **kw):
        seen["url"] = url
        seen.update(kw)
        if exc:
            raise exc
        return httpx.Response(status, json=body if body is not None else {}, request=httpx.Request("POST", url))

    monkeypatch.setattr(stt.httpx, "post", post)
    monkeypatch.setattr(config, "STT_ENGINE", "elevenlabs")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", key)
    monkeypatch.setattr(config, "ELEVENLABS_STT_MODEL", "scribe_v2")
    return seen


def test_scribe_request_shape(monkeypatch):
    seen = fake_scribe(monkeypatch, body={"text": "Hi.", "language_code": "eng"})
    stt.transcribe_audio(b"audio-bytes")
    assert seen["url"] == "https://api.elevenlabs.io/v1/speech-to-text"
    assert seen["headers"] == {"xi-api-key": "xi-test"}
    assert seen["data"] == {"model_id": "scribe_v2", "tag_audio_events": "false", "timestamps_granularity": "none"}
    assert "language_code" not in seen["data"]  # auto-detect: people switch language between questions
    assert seen["files"]["file"][1] == b"audio-bytes"


def test_scribe_arabic(monkeypatch):
    fake_scribe(monkeypatch, body={"text": " وش أخبارك؟ ", "language_code": "ara"})
    assert stt.transcribe_audio(b"x") == stt.Transcript("وش أخبارك؟", "ar")


def test_scribe_english(monkeypatch):
    fake_scribe(monkeypatch, body={"text": "Hi there.", "language_code": "eng"})
    assert stt.transcribe_audio(b"x") == stt.Transcript("Hi there.", "en")


def test_scribe_two_letter_codes(monkeypatch):
    fake_scribe(monkeypatch, body={"text": "", "language_code": "ar"})
    assert stt.transcribe_audio(b"x").language == "ar"


def test_scribe_mixed_arabic_english_counts_as_arabic(monkeypatch):
    fake_scribe(monkeypatch, body={"text": "عندي meeting بكرة", "language_code": "eng"})
    assert stt.transcribe_audio(b"x") == stt.Transcript("عندي meeting بكرة", "ar")


def test_scribe_other_language_defaults_to_english(monkeypatch):
    fake_scribe(monkeypatch, body={"text": "bonjour", "language_code": "fra"})
    assert stt.transcribe_audio(b"x") == stt.Transcript("bonjour", "en")


def test_scribe_http_error_is_a_service_error(monkeypatch):
    fake_scribe(monkeypatch, status=401, body={"detail": "nope"})
    with pytest.raises(stt.SpeechServiceError):
        stt.transcribe_audio(b"x")


def test_scribe_network_error_is_a_service_error(monkeypatch):
    fake_scribe(monkeypatch, exc=httpx.ConnectError("refused"))
    with pytest.raises(stt.SpeechServiceError):
        stt.transcribe_audio(b"x")


def test_scribe_does_not_load_whisper(monkeypatch):
    fake_scribe(monkeypatch)
    monkeypatch.setattr(stt, "get_whisper", lambda: (_ for _ in ()).throw(AssertionError("loaded whisper")))
    assert stt.get_transcriber() is not None


def test_scribe_needs_a_key(monkeypatch):
    fake_scribe(monkeypatch, key="")
    with pytest.raises(stt.NoKeyError):
        stt.get_transcriber()


def test_route_no_key_message(monkeypatch):
    fake_scribe(monkeypatch, key="")
    r = client.post("/api/transcribe", content=b"audio")
    assert r.status_code == 503
    assert r.json() == {"detail": "ElevenLabs API key missing. Add ELEVENLABS_API_KEY to .env and restart."}


def test_route_service_error_is_502(monkeypatch):
    fake_scribe(monkeypatch, status=500, body={})
    r = client.post("/api/transcribe", content=b"audio")
    assert r.status_code == 502 and r.json() == {"detail": "Speech service unavailable. Try again."}


# ---- route errors (kept from before, re-pointed at the new function names) ----
def test_transcribe_too_big(monkeypatch):
    monkeypatch.setattr(stt, "MAX_AUDIO_BYTES", 10)
    assert client.post("/api/transcribe", content=b"x" * 11).status_code == 413


def test_transcribe_too_big_by_content_length(monkeypatch):
    monkeypatch.setattr(stt, "MAX_AUDIO_BYTES", 10)
    monkeypatch.setattr(stt, "get_transcriber", lambda: None)
    r = client.post("/api/transcribe", content=b"x", headers={"Content-Length": "11"})
    assert r.status_code == 413


def test_transcribe_bad_audio(monkeypatch):
    monkeypatch.setattr(stt, "get_transcriber", lambda: None)

    def boom(d):
        raise ValueError("invalid data")

    monkeypatch.setattr(stt, "transcribe_audio", boom)
    r = client.post("/api/transcribe", content=b"garbage")
    assert r.status_code == 400 and r.json() == {"detail": "Could not decode audio."}


def test_transcribe_model_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("download failed")

    monkeypatch.setattr(stt, "get_transcriber", boom)
    r = client.post("/api/transcribe", content=b"audio")
    assert r.status_code == 503 and r.json() == {"detail": "Speech model unavailable."}
