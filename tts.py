import functools
import io
import json
import wave
from pathlib import Path
from typing import Callable, Protocol

import config
import dialects

VOICES_DIR = Path(__file__).parent / "voices"


class VoiceEngine(Protocol):
    def synthesize(self, text: str) -> bytes:  # WAV bytes
        ...


def _patch_onnxruntime_providers():
    """onnxruntime builds with more than one execution provider require an explicit
    `providers` argument; piper's Arabic tashkeel diacritizer omits it (so it crashes on
    onnxruntime 1.16, whose macOS build enables AzureExecutionProvider). Default to CPU."""
    import onnxruntime as ort

    if getattr(ort, "_rawi_providers_patched", False):
        return
    base = ort.InferenceSession

    class PatchedInferenceSession(base):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("providers", ["CPUExecutionProvider"])
            super().__init__(*args, **kwargs)

    ort.InferenceSession = PatchedInferenceSession
    ort._rawi_providers_patched = True


def _piper_engine(spec: dict) -> VoiceEngine:
    _patch_onnxruntime_providers()
    from piper import PiperVoice  # lazy: importing tts must not load piper or any voice

    voice = PiperVoice.load(VOICES_DIR / f"{spec['model']}.onnx")

    class Engine:
        def synthesize(self, text: str) -> bytes:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            return buf.getvalue()

    return Engine()


def _elevenlabs_engine(spec: dict) -> VoiceEngine:
    import httpx  # lazy: importing tts stays light

    voice_id = spec["voice_id"]
    model = spec.get("model_id", "eleven_multilingual_v2")

    class Engine:
        def synthesize(self, text: str) -> bytes:
            r = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": config.ELEVENLABS_API_KEY, "Content-Type": "application/json"},
                # output_format is a query parameter: in the JSON body it is ignored and ElevenLabs
                # answers with MP3, which would be wrapped below as PCM and play as loud noise
                params={"output_format": "pcm_24000"},
                json={"text": text, "model_id": model},
                timeout=httpx.Timeout(30, connect=5),
            )
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            if content_type and not content_type.startswith(("audio/pcm", "application/octet-stream")):
                raise ValueError(f"expected raw PCM from ElevenLabs, got {content_type}")
            # wrap the returned 24 kHz 16-bit mono PCM into a WAV for the page
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(r.content)
            return buf.getvalue()

    return Engine()


ENGINES: dict[str, Callable[[dict], VoiceEngine]] = {
    "piper": _piper_engine,
    "elevenlabs": _elevenlabs_engine,
}


def load_voice_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {"en": {"engine": "piper", "model": config.PIPER_VOICE}}
    data = json.loads(path.read_text())
    out = {}
    for dialect, spec in data.items():
        if dialect not in dialects.DIALECTS:
            continue
        if spec.get("engine") not in ENGINES:
            raise ValueError(f"unknown engine for {dialect}: {spec.get('engine')!r}")
        out[dialect] = spec
    return out


def _voice_map() -> dict[str, dict]:
    return load_voice_map(Path(config.VOICES_FILE))


@functools.lru_cache(maxsize=8)
def get_engine(dialect: str) -> VoiceEngine:
    spec = _voice_map().get(dialect)
    if spec is None:
        raise KeyError(dialect)
    return ENGINES[spec["engine"]](spec)


def synthesize(text: str, dialect: str) -> bytes:
    return get_engine(dialect).synthesize(text)


def available_dialects() -> list[str]:
    out = []
    for dialect, spec in _voice_map().items():
        engine = spec["engine"]
        if engine == "piper" and (VOICES_DIR / f"{spec['model']}.onnx").exists():
            out.append(dialect)
        elif engine == "elevenlabs" and config.ELEVENLABS_API_KEY:
            out.append(dialect)
    return out
