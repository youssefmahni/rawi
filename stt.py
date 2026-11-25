import functools
import io
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

import config
import dialects

MAX_AUDIO_BYTES = 25 * 1024 * 1024

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
NO_KEY_MESSAGE = "ElevenLabs API key missing. Add ELEVENLABS_API_KEY to .env and restart."
# Scribe reports ISO 639-3 codes ("ara", "eng"); the dialect variants and the two-letter form are listed too.
ARABIC_CODES = {"ar", "ara", "arb", "ary", "arz", "apc", "ars", "acm", "afb", "ayl", "aeb", "apd", "ajp", "acw", "aec", "abv"}


class NoKeyError(RuntimeError):
    """The cloud speech engine is selected but no ElevenLabs key is configured."""


class SpeechServiceError(RuntimeError):
    """The cloud speech service could not be reached or refused the request."""


@dataclass(frozen=True)
class Transcript:
    text: str
    language: Literal["en", "ar"]


class Transcriber(Protocol):
    def transcribe(self, audio: bytes) -> Transcript: ...


def _device() -> str:
    device = config.STT_DEVICE
    if device == "auto":
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    return device


@functools.cache
def get_whisper():
    from faster_whisper import WhisperModel  # lazy: keeps import of app fast, model loads on first use

    device = _device()
    compute = config.STT_COMPUTE or ("int8" if device == "cpu" else "float16")
    return WhisperModel(config.STT_MODEL, device=device, compute_type=compute,
                        cpu_threads=config.STT_CPU_THREADS)


def transcribe_audio(data: bytes) -> Transcript:
    if config.STT_ENGINE == "elevenlabs":
        return _transcribe_scribe(data)
    return _transcribe_whisper(data)


def _language_of(code: str, text: str) -> Literal["en", "ar"]:
    # Arabic script wins: a sentence that mixes English words into Arabic is answered in Arabic.
    if dialects.has_arabic(text) or (code or "").lower() in ARABIC_CODES:
        return "ar"
    return "en"


def _transcribe_scribe(data: bytes) -> Transcript:
    """ElevenLabs Scribe: audio goes to the cloud. Language is auto-detected for every question."""
    try:
        r = httpx.post(
            SCRIBE_URL,
            headers={"xi-api-key": config.ELEVENLABS_API_KEY},
            data={"model_id": config.ELEVENLABS_STT_MODEL, "tag_audio_events": "false",
                  "timestamps_granularity": "none"},
            files={"file": ("audio", data, "application/octet-stream")},
            timeout=httpx.Timeout(60, connect=5),
        )
        r.raise_for_status()
        body = r.json()
    except (httpx.HTTPError, ValueError) as exc:  # HTTPError covers status errors and network errors
        raise SpeechServiceError(type(exc).__name__) from None  # type name only: never echo request details
    text = (body.get("text") or "").strip()
    return Transcript(text, _language_of(body.get("language_code") or "", text))


def _transcribe_whisper(data: bytes) -> Transcript:
    # beam_size=1 (greedy): warm transcription of a 2.4 s question drops from ~4.6 s to ~3.3 s vs the default 5
    segments, info = get_whisper().transcribe(io.BytesIO(data), language=None, vad_filter=True, beam_size=1)
    text = "".join(seg.text for seg in segments).strip()
    if info.language in ("en", "ar"):
        return Transcript(text, info.language)
    # Whisper misdetected the language (common for Darija: fr/ko/it/pt/…). Redo it restricted
    # to whichever of en/ar is more likely, with ties going to Arabic.
    probs = dict(info.all_language_probs or [])
    language = "ar" if probs.get("ar", 0.0) >= probs.get("en", 0.0) else "en"
    segments, _ = get_whisper().transcribe(io.BytesIO(data), language=language, vad_filter=True, beam_size=1)
    return Transcript("".join(seg.text for seg in segments).strip(), language)


class _Transcriber:
    def transcribe(self, audio: bytes) -> Transcript:
        return transcribe_audio(audio)


@functools.cache
def _whisper_transcriber() -> Transcriber:
    get_whisper()  # loading here lets the route report a 503 on a download failure rather than a 400
    return _Transcriber()


def get_transcriber() -> Transcriber:
    """The cloud engine loads nothing but needs its key, checked on every call so that a removed or
    missing key is reported (503) instead of failing later as a bad request. The local engine loads
    and caches its model."""
    if config.STT_ENGINE == "elevenlabs":
        if not config.ELEVENLABS_API_KEY:
            raise NoKeyError(NO_KEY_MESSAGE)
        return _Transcriber()
    return _whisper_transcriber()
