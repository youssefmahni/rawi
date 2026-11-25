import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse
from pydantic import BaseModel, Field

import config
import dialects
import llm
import stt
import tts

log = logging.getLogger(__name__)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---- request guards ----
class SameOriginMiddleware:
    """Reject (403) non-GET requests sent by another site: Sec-Fetch-Site: cross-site, or an Origin
    that is not this app. Requests without these headers (curl, tests) pass."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] != "GET":
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            origin = headers.get("origin")
            allowed = config.allowed_origins()
            if headers.get("sec-fetch-site") == "cross-site" or (origin is not None and origin not in allowed):
                await PlainTextResponse("Cross-site request rejected.", status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ---- routes ----
@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await llm.close_deepseek()


app = FastAPI(lifespan=lifespan)
app.add_middleware(SameOriginMiddleware)
# DNS rebinding guard: a hostile page rebound to 127.0.0.1 still sends its own Host name.
# Added last so it runs first.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.ALLOWED_HOSTS)
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(config.BASE_DIR / "static" / "index.html")


@app.get("/api/health")
async def health():
    available = set(tts.available_dialects())
    return {"deepseek_key": bool(config.DEEPSEEK_API_KEY),
            "voices": {d: (d in available) for d in dialects.DIALECTS}}


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = []
    language: Literal["en", "ar"] | None = None

    model_config = {"str_strip_whitespace": True}


async def ask_events(body: AskBody) -> AsyncIterator[str]:
    """One DeepSeek turn, streamed as dialect → delta… → done (or error)."""
    lang = body.language or dialects.language_of(dialects.guess_dialect(body.question))
    if not config.DEEPSEEK_API_KEY:
        yield sse("error", {"message": llm.NO_KEY, "kind": "nokey", "language": lang})
        return
    messages = llm.build_messages(body.question, [t.model_dump() for t in body.history])
    parser = llm.DialectTagParser()
    # holds the tag until the first sentence is in, then checks it against the words (and the question's)
    gate = dialects.DialectGate(question=body.question)
    t0 = time.perf_counter()
    first_delta_logged = False

    def release(dialect, pieces) -> list[str]:
        nonlocal first_delta_logged
        events = [sse("dialect", {"dialect": dialect})] if dialect else []
        for piece in pieces:
            if not first_delta_logged:
                first_delta_logged = True
                log.info("timing llm_first_ms=%.0f", (time.perf_counter() - t0) * 1000)
            events.append(sse("delta", {"text": piece}))
        return events

    try:
        async for chunk in llm.stream_turn(messages):
            tag, text = parser.feed(chunk)
            for event in release(*gate.push(tag, text)):
                yield event
    except llm.OfflineError:
        for event in release(*gate.flush()):  # text that was already received stays
            yield event
        yield sse("error", {"message": llm.OFFLINE[lang], "kind": "offline", "language": lang})
        return
    except Exception as exc:
        log.error("DeepSeek request failed: %s", type(exc).__name__, exc_info=False)
        for event in release(*gate.flush()):
            yield event
        yield sse("error", {"message": llm.LLM_FAILED, "kind": "failed", "language": lang})
        return
    tag, text = parser.finish()
    for event in release(*gate.push(tag, text)):
        yield event
    for event in release(*gate.flush()):
        yield event
    yield sse("done", {})


@app.post("/api/ask")
async def ask(body: AskBody):
    return StreamingResponse(ask_events(body), media_type="text/event-stream")


@app.post("/api/transcribe")
async def transcribe(request: Request):
    try:
        declared = int(request.headers.get("content-length", "0"))
    except ValueError:
        declared = 0
    if declared > stt.MAX_AUDIO_BYTES:  # refuse before reading the body
        raise HTTPException(status_code=413, detail="Audio too large.")
    data = await request.body()
    if not data:
        return {"text": "", "language": "en"}
    if len(data) > stt.MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large.")
    try:
        await run_in_threadpool(stt.get_transcriber)  # load (cached) separately so load failures are not "bad audio"
    except stt.NoKeyError:
        raise HTTPException(status_code=503, detail=stt.NO_KEY_MESSAGE)
    except Exception:
        log.exception("Whisper model load failed")
        raise HTTPException(status_code=503, detail="Speech model unavailable.")
    try:
        t0 = time.perf_counter()
        result = await run_in_threadpool(stt.transcribe_audio, data)
        log.info("timing stt_ms=%.0f", (time.perf_counter() - t0) * 1000)
    except stt.SpeechServiceError as exc:
        log.error("Speech service failed: %s", exc)  # the error carries only an exception type name
        raise HTTPException(status_code=502, detail="Speech service unavailable. Try again.")
    except Exception:
        log.exception("Transcription failed")
        raise HTTPException(status_code=400, detail="Could not decode audio.")
    return {"text": result.text, "language": result.language}


class SpeakBody(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    dialect: dialects.Dialect

    model_config = {"str_strip_whitespace": True}


@app.post("/api/speak")
async def speak(body: SpeakBody):
    try:
        await run_in_threadpool(tts.get_engine, body.dialect)  # load (cached) separately so load failures are not synthesis errors
    except KeyError:
        raise HTTPException(status_code=503, detail="No voice for this dialect.")
    except Exception:
        log.exception("Voice load failed")
        raise HTTPException(status_code=503, detail="Voice unavailable.")
    try:
        audio = await run_in_threadpool(tts.synthesize, body.text, body.dialect)
    except Exception:
        log.exception("Speech synthesis failed")
        raise HTTPException(status_code=500, detail="Speech synthesis failed.")
    return Response(audio, media_type="audio/wav")


def preload_models() -> None:
    """Load the transcriber and every configured voice before serving. Failures are logged, not
    fatal: the routes retry the load and return 503 if it still fails."""
    if config.STT_ENGINE == "elevenlabs":
        print(f"Using ElevenLabs Scribe ({config.ELEVENLABS_STT_MODEL}) for speech-to-text", flush=True)
    else:
        print(f"Loading Whisper {config.STT_MODEL}\u2026", flush=True)
    try:
        stt.get_transcriber()
    except Exception:
        log.exception("Whisper preload failed")
    for dialect in tts.available_dialects():
        print(f"Loading voice {dialect}\u2026", flush=True)
        try:
            tts.get_engine(dialect)
        except Exception:
            log.exception("Voice preload failed for %s", dialect)


if __name__ == "__main__":
    preload_models()
    uvicorn.run(app, host=config.HOST, port=config.PORT, **config.uvicorn_ssl_kwargs())
