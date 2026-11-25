# Whisper Voice Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app where you ask a question out loud in English and hear a short answer grounded in the ciso-rag knowledge base, shown as a HIG-styled chat.

**Architecture:** One FastAPI process (`app.py`) bound to 127.0.0.1 serves a static page and five JSON/stream endpoints. Speech-to-text is faster-whisper `small.en` on CPU, retrieval is the existing ciso-rag HTTP API, the answer streams from DeepSeek (OpenAI-compatible) as Server-Sent Events, and Piper turns each finished sentence into WAV. The page is plain HTML/CSS/JS with no build step.

**Tech Stack:** Python 3.13, FastAPI + uvicorn, faster-whisper 1.2.1, piper-tts 1.8.0, openai SDK (pointed at DeepSeek), httpx, python-dotenv, pytest; vanilla JS (ES modules), `node --test` for the one JS module with logic.

**Spec:** `docs/superpowers/specs/2026-11-15-whisper-voice-assistant-design.md`

## Global Constraints

- Server binds to `127.0.0.1` only. Never `0.0.0.0`.
- `DEEPSEEK_API_KEY` is read from `.env` only. Never sent to the page, logged, or committed. `.env` is git-ignored; `.env.example` is committed.
- Config keys and defaults (verbatim): `DEEPSEEK_API_KEY=` · `DEEPSEEK_MODEL=deepseek-chat` · `CISO_RAG_URL=http://127.0.0.1:18760` · `WHISPER_MODEL=small.en` · `PIPER_VOICE=en_US-lessac-medium` · `PORT=8765`.
- DeepSeek base URL: `https://api.deepseek.com`.
- ciso-rag: `GET /api/health`, `POST /api/v1/search` with `{query, limit, rerank: true}`. No auth header. Labels: `PUBLIC`, `INTERNAL`, `RESTRICTED`, `CONFIDENTIAL` (exact, uppercase).
- Whisper model and Piper voice load lazily on first use (cached). Importing `app` must not load any model, so tests run fast.
- English only. No saved chat history, no accounts, no wake word.
- Accepted latency is 3–6 s from send to the first spoken word. No streaming transcription or voice-activity detection.
- Every git commit message ends with: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **SSH tunnel down** (ciso-rag unreachable): `/api/health` reports `"rag": "down"` and `/api/ask` sends one `error` event "Knowledge base unreachable. Is the SSH tunnel up?" with no crash. Tests are in Task 1 and Task 3.
2. **DeepSeek key missing or DeepSeek failing mid-answer:** the server still starts, `/api/ask` sends a clear `error` event, and any partial answer stays on screen. Tests are in Task 3.
3. **Silence or an accidental mic tap:** transcription returns `""` and the UI shows "Didn't catch that" instead of asking an empty question. Tests are in Task 4; UI behaviour is in Task 7.
4. **Regulatory numbering and abbreviations in answers** ("ECC 2-2-3", "v1.0", "e.g."), plus stray markdown: these must not split speech mid-sentence or be read aloud as symbols. Tests are in Task 6.
5. **Untrusted `voice` parameter** (e.g. `../../etc/passwd`): it must be rejected with 400 and never used as a path. Test is in Task 5.

---

## File Structure

```
app.py                 config, ciso-rag client, prompt, DeepSeek stream, Whisper, Piper, routes
static/index.html      page markup
static/app.css         HIG tokens + layout, light/dark
static/app.js          mic, SSE client, audio queue, settings, health
static/speech.js       SentenceBuffer + cleanForSpeech (pure, testable)
tests/test_app.py      server tests (models and network faked)
tests/speech.test.mjs  node --test for speech.js
voices/                Piper .onnx + .onnx.json (git-ignored, downloaded)
requirements.txt
.env.example
.gitignore
README.md
```

`app.py` stays one file, as the spec says, and should land around 250 lines. Section it with comments in this order: config, ciso-rag, prompt, DeepSeek, Whisper, Piper, routes.

---

### Task 1: Skeleton, config and health endpoint

**Files:**
- Create: `app.py`, `static/index.html` (placeholder `<h1>CISO Assistant</h1>`), `requirements.txt`, `.env.example`, `.gitignore`, `tests/test_app.py`

**Interfaces:**
- Produces:
  - Module constants `DEEPSEEK_API_KEY: str`, `DEEPSEEK_MODEL: str`, `CISO_RAG_URL: str`, `WHISPER_MODEL: str`, `PIPER_VOICE: str`, `PORT: int`, loaded via `python-dotenv` + `os.getenv` with the Global Constraints defaults. Tests monkeypatch these.
  - `async def rag_get(path: str) -> dict`: GET `CISO_RAG_URL + path`, timeout 10 s, `raise_for_status()`, returns JSON.
  - `async def rag_post(path: str, payload: dict) -> dict`: same, POST JSON, timeout 30 s.
  - `async def rag_health() -> dict`: `rag_get("/api/health")`.
  - `app: FastAPI`. `GET /` serves `static/index.html`, and `/static` is mounted.
  - `GET /api/health` returns `{"rag": "ok"|"down", "indexed_documents": int|None, "deepseek_key": bool}`. Any `httpx.HTTPError` means `"down"`.
  - `if __name__ == "__main__": uvicorn.run(app, host="127.0.0.1", port=PORT)`

- [ ] **Step 1: Init repo and environment**

```bash
cd /home/youssef/Desktop/whisper && git init
python3 -m venv .venv
```
`requirements.txt`:
```
fastapi
uvicorn
httpx
python-dotenv
openai
faster-whisper==1.2.1
piper-tts==1.8.0
pytest
```
`.gitignore`: `.venv/`, `.env`, `voices/`, `__pycache__/`, `.pytest_cache/`. `.env.example` holds the six keys from Global Constraints with the key value empty.

Run: `.venv/bin/pip install -r requirements.txt`
Expected: ends with `Successfully installed ...` and no errors.

- [ ] **Step 2: Write the failing tests**

```python
import httpx
from fastapi.testclient import TestClient
import app as app_module

client = TestClient(app_module.app)

def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]

def test_health_ok(monkeypatch):
    async def fake_get(path):
        assert path == "/api/health"
        return {"status": "healthy", "indexed_documents": 1208}
    monkeypatch.setattr(app_module, "rag_get", fake_get)
    monkeypatch.setattr(app_module, "DEEPSEEK_API_KEY", "sk-test")
    assert client.get("/api/health").json() == {"rag": "ok", "indexed_documents": 1208, "deepseek_key": True}

def test_health_rag_down(monkeypatch):
    async def boom(path):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(app_module, "rag_get", boom)
    monkeypatch.setattr(app_module, "DEEPSEEK_API_KEY", "")
    assert client.get("/api/health").json() == {"rag": "down", "indexed_documents": None, "deepseek_key": False}
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 4: Implement the Task 1 interfaces in `app.py`**

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit** (include the existing `docs/` spec and this plan)

```bash
git add . && git commit -m "feat: app skeleton, config and health endpoint"
```

---

### Task 2: Retrieval filter, search and prompt

**Files:**
- Modify: `app.py` (ciso-rag and prompt sections)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `rag_post` (Task 1).
- Produces:
  - `KNOWN_LEVELS = {"PUBLIC", "INTERNAL", "RESTRICTED", "CONFIDENTIAL"}`, `SENSITIVE = {"RESTRICTED", "CONFIDENTIAL"}`
  - `def filter_passages(results: list[dict], classification: str) -> list[dict]`: `classification` is `"all"` or `"exclude_sensitive"`. For `"all"`, return the first 5. For `"exclude_sensitive"`, keep only items whose `classification` is in `KNOWN_LEVELS - SENSITIVE` (fail-closed), then return the first 5.
  - `async def search_rag(question: str, classification: str) -> list[dict]`: calls `rag_post("/api/v1/search", {"query": question, "limit": 5 if all else 10, "rerank": True})` and returns `filter_passages(data.get("results", []), classification)`.
  - `SYSTEM_PROMPT: str` (exact copy below)
  - `def build_messages(question: str, history: list[dict], passages: list[dict]) -> list[dict]`: `[system] + history[-6:] + [user]`. The user content is the numbered passages, each as `[n] {file_name} — {header_path}\n{text}`, separated by blank lines, then a blank line, then `Question: {question}`.

`SYSTEM_PROMPT`:
```
You are a voice assistant for a CISO knowledge base. Answer only from the numbered passages in the user's message. If they do not contain the answer, say: "I couldn't find that in the knowledge base." Reply in 2 to 4 short sentences of plain spoken English. No markdown, no lists, no headings, and do not read out document names or passage numbers.
```

- [ ] **Step 1: Write the failing tests**

```python
P = lambda c, n="doc.md": {"file_name": n, "header_path": "A > B", "text": "t", "classification": c}

def test_filter_all_keeps_first_five():
    rs = [P("RESTRICTED", f"d{i}") for i in range(7)]
    assert [r["file_name"] for r in app_module.filter_passages(rs, "all")] == ["d0", "d1", "d2", "d3", "d4"]

def test_filter_exclude_is_fail_closed():
    rs = [P("PUBLIC", "a"), P("RESTRICTED", "b"), P("CONFIDENTIAL", "c"),
          P(None, "d"), P("internal", "e"), P("SECRET", "f"), P("INTERNAL", "g")]
    assert [r["file_name"] for r in app_module.filter_passages(rs, "exclude_sensitive")] == ["a", "g"]

def test_search_rag_payload(monkeypatch):
    seen = []
    async def fake_post(path, payload):
        seen.append((path, payload)); return {"results": [P("INTERNAL")]}
    monkeypatch.setattr(app_module, "rag_post", fake_post)
    import asyncio
    asyncio.run(app_module.search_rag("mfa?", "all"))
    asyncio.run(app_module.search_rag("mfa?", "exclude_sensitive"))
    assert seen == [("/api/v1/search", {"query": "mfa?", "limit": 5, "rerank": True}),
                    ("/api/v1/search", {"query": "mfa?", "limit": 10, "rerank": True})]

def test_build_messages_shape():
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(8)]
    p = {"file_name": "P-03-Access-Control-v1.0.md", "header_path": "Access Control Policy > 10. References",
         "text": "M1032: Multi-factor Authentication", "classification": "INTERNAL"}
    msgs = app_module.build_messages("Is MFA required?", hist, [p])
    assert msgs[0] == {"role": "system", "content": app_module.SYSTEM_PROMPT}
    assert [m["content"] for m in msgs[1:-1]] == ["2", "3", "4", "5", "6", "7"]
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == ("[1] P-03-Access-Control-v1.0.md — Access Control Policy > 10. References\n"
                                   "M1032: Multi-factor Authentication\n\nQuestion: Is MFA required?")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: the 4 new tests FAIL with `AttributeError: module 'app' has no attribute 'filter_passages'` (and similar)

- [ ] **Step 3: Implement `filter_passages`, `search_rag`, `SYSTEM_PROMPT` and `build_messages` in `app.py`**

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py && git commit -m "feat: ciso-rag search with fail-closed classification filter and prompt"
```

---

### Task 3: `/api/ask` streaming answer with guards

**Files:**
- Modify: `app.py` (DeepSeek section and routes)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `search_rag`, `build_messages` (Task 2), `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`.
- Produces:
  - `async def stream_answer(messages: list[dict]) -> AsyncIterator[str]`: `openai.AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")`, `chat.completions.create(model=DEEPSEEK_MODEL, messages=..., stream=True)`, yielding non-empty `choices[0].delta.content`.
  - `def sse(event: str, data) -> str`: returns `f"event: {event}\ndata: {json.dumps(data)}\n\n"`.
  - Constants (exact copy):
    - `NOT_FOUND = "I couldn't find that in the knowledge base."`
    - `RAG_DOWN = "Knowledge base unreachable. Is the SSH tunnel up?"`
    - `NO_KEY = "DeepSeek API key missing. Add DEEPSEEK_API_KEY to .env and restart."`
    - `LLM_FAILED = "DeepSeek request failed. Try again."`
  - `POST /api/ask`, with a pydantic body: `question: str` (strip, 1–2000 chars), `history: list[{role: Literal["user","assistant"], content: str}] = []`, `classification: Literal["all","exclude_sensitive"] = "all"`. The response is `StreamingResponse(media_type="text/event-stream")` with this event sequence:
    1. Key empty → `error {"message": NO_KEY}`, then stop.
    2. `search_rag` raises `httpx.HTTPError` → `error {"message": RAG_DOWN}`, then stop.
    3. `sources` → list of `{file_name, header_path, classification, text}`.
    4. No passages → `delta {"text": NOT_FOUND}`, `done {}`, without calling DeepSeek.
    5. Otherwise, one `delta {"text": chunk}` per chunk, then `done {}`.
    6. An exception while streaming → log it server-side (never the key) and send `error {"message": LLM_FAILED}`.

- [ ] **Step 1: Write the failing tests**

```python
import json

def events(r):
    out = []
    for block in r.text.strip().split("\n\n"):
        lines = dict(l.split(": ", 1) for l in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out

def ask(**body):
    return client.post("/api/ask", json={"question": "Is MFA required?", **body})

@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setattr(app_module, "DEEPSEEK_API_KEY", "sk-test")

def test_ask_happy_path(monkeypatch, keyed):
    async def s(q, c): return [P("INTERNAL")]
    async def llm(msgs):
        for t in ["MFA is required. ", "For admins."]: yield t
    monkeypatch.setattr(app_module, "search_rag", s)
    monkeypatch.setattr(app_module, "stream_answer", llm)
    ev = events(ask())
    assert [e for e, _ in ev] == ["sources", "delta", "delta", "done"]
    assert ev[0][1] == [{"file_name": "doc.md", "header_path": "A > B", "classification": "INTERNAL", "text": "t"}]
    assert ev[1][1] == {"text": "MFA is required. "}

def test_ask_no_results_skips_llm(monkeypatch, keyed):
    async def s(q, c): return []
    async def llm(msgs):
        raise AssertionError("must not call DeepSeek"); yield
    monkeypatch.setattr(app_module, "search_rag", s)
    monkeypatch.setattr(app_module, "stream_answer", llm)
    assert events(ask()) == [("sources", []), ("delta", {"text": app_module.NOT_FOUND}), ("done", {})]

def test_ask_rag_down(monkeypatch, keyed):
    async def s(q, c): raise httpx.ConnectError("refused")
    monkeypatch.setattr(app_module, "search_rag", s)
    assert events(ask()) == [("error", {"message": app_module.RAG_DOWN})]

def test_ask_no_key(monkeypatch):
    monkeypatch.setattr(app_module, "DEEPSEEK_API_KEY", "")
    assert events(ask()) == [("error", {"message": app_module.NO_KEY})]

def test_ask_llm_fails_mid_stream(monkeypatch, keyed):
    async def s(q, c): return [P("INTERNAL")]
    async def llm(msgs):
        yield "Partial"; raise RuntimeError("boom")
    monkeypatch.setattr(app_module, "search_rag", s)
    monkeypatch.setattr(app_module, "stream_answer", llm)
    assert [e for e, _ in events(ask())] == ["sources", "delta", "error"]
    assert events(ask())[-1][1] == {"message": app_module.LLM_FAILED}

def test_ask_validation():
    assert client.post("/api/ask", json={"question": "  "}).status_code == 422
    assert ask(classification="bogus").status_code == 422
```
(Add `import pytest` at the top of the file.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_app.py -k ask -v`
Expected: FAIL (404 Not Found on `/api/ask`)

- [ ] **Step 3: Implement `stream_answer`, `sse`, the four constants and `POST /api/ask` in `app.py`**

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py && git commit -m "feat: /api/ask streams grounded DeepSeek answer over SSE"
```

---

### Task 4: `/api/transcribe` with Whisper

**Files:**
- Modify: `app.py` (Whisper section and routes)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `WHISPER_MODEL`.
- Produces:
  - `@functools.cache def get_whisper() -> WhisperModel`: `WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")`
  - `def transcribe_audio(data: bytes) -> str`: `segments, _ = get_whisper().transcribe(io.BytesIO(data), language="en", vad_filter=True)`, then the segment texts joined with `""` and stripped.
  - `MAX_AUDIO_BYTES = 25 * 1024 * 1024`
  - `POST /api/transcribe` takes the raw request body (any audio content type the browser's MediaRecorder sends, e.g. `audio/webm`) and returns `{"text": str}`:
    - empty body → `{"text": ""}` without loading the model
    - over `MAX_AUDIO_BYTES` → 413
    - decode/transcribe exception → 400 `{"detail": "Could not decode audio."}`
  - Run `transcribe_audio` with `run_in_threadpool` so a transcription doesn't block the event loop.

- [ ] **Step 1: Write the failing tests**

```python
class Seg:
    def __init__(self, text): self.text = text

def fake_model(monkeypatch, segs=None, exc=None):
    calls = {}
    class M:
        def transcribe(self, audio, **kw):
            calls.update(kw)
            if exc: raise exc
            return iter(segs or []), None
    monkeypatch.setattr(app_module, "get_whisper", lambda: M())
    return calls

def test_transcribe_text(monkeypatch):
    calls = fake_model(monkeypatch, [Seg(" Is MFA"), Seg(" required?")])
    r = client.post("/api/transcribe", content=b"webmbytes", headers={"Content-Type": "audio/webm"})
    assert r.json() == {"text": "Is MFA required?"}
    assert calls["language"] == "en" and calls["vad_filter"] is True

def test_transcribe_silence_is_empty(monkeypatch):
    fake_model(monkeypatch, [])
    assert client.post("/api/transcribe", content=b"x").json() == {"text": ""}

def test_transcribe_empty_body_skips_model(monkeypatch):
    monkeypatch.setattr(app_module, "get_whisper", lambda: (_ for _ in ()).throw(AssertionError("loaded")))
    assert client.post("/api/transcribe", content=b"").json() == {"text": ""}

def test_transcribe_too_big(monkeypatch):
    monkeypatch.setattr(app_module, "MAX_AUDIO_BYTES", 10)
    assert client.post("/api/transcribe", content=b"x" * 11).status_code == 413

def test_transcribe_bad_audio(monkeypatch):
    fake_model(monkeypatch, exc=ValueError("invalid data"))
    r = client.post("/api/transcribe", content=b"garbage")
    assert r.status_code == 400 and r.json() == {"detail": "Could not decode audio."}
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_app.py -k transcribe -v`
Expected: FAIL (404 on `/api/transcribe`)

- [ ] **Step 3: Implement `get_whisper`, `transcribe_audio`, `MAX_AUDIO_BYTES` and `POST /api/transcribe` in `app.py`**

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: 18 passed

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py && git commit -m "feat: /api/transcribe with faster-whisper small.en"
```

---

### Task 5: `/api/speak` and `/api/voices` with Piper, plus a real round trip

**Files:**
- Modify: `app.py` (Piper section and routes)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `PIPER_VOICE`, `transcribe_audio` (Task 4) for the round-trip check.
- Produces:
  - `VOICES_DIR = Path(__file__).parent / "voices"`
  - `def list_voices() -> list[str]`: sorted stems of `VOICES_DIR/*.onnx` that have a matching `.onnx.json`.
  - `@functools.lru_cache(maxsize=4) def get_voice(name: str) -> PiperVoice`: `PiperVoice.load(VOICES_DIR / f"{name}.onnx")`
  - `def synthesize(text: str, voice: str) -> bytes`: writes WAV into `io.BytesIO` with the `wave` module and `get_voice(voice).synthesize_wav(text, wav_file)`. Confirm that method name in Step 1.
  - `GET /api/voices` returns `{"voices": list_voices(), "default": PIPER_VOICE}`.
  - `POST /api/speak` takes body `text: str` (1–1000 chars after strip) and `voice: str | None = None` (`None` means `PIPER_VOICE`), and returns `Response(media_type="audio/wav")`:
    - no voices installed → 503 `{"detail": "No Piper voice installed. See README."}`
    - `voice` not in `list_voices()` → 400 `{"detail": "Unknown voice."}` (it's never joined into a path before this check)
    - synthesis runs via `run_in_threadpool`

- [ ] **Step 1: Download the default voice and confirm the Piper API**

```bash
.venv/bin/python -m piper.download_voices en_US-lessac-medium --data-dir voices
.venv/bin/python -c "from piper import PiperVoice; print(PiperVoice.synthesize_wav.__doc__ or 'ok')"
```
Expected: `voices/en_US-lessac-medium.onnx` and `.onnx.json` exist, and the second command prints without `AttributeError`. If `download_voices` or `synthesize_wav` is missing in 1.8.0, check `python -m piper --help` / `help(PiperVoice)` for the 1.8 equivalent and use that. The voice files can also be fetched directly from `https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx` (and `.onnx.json`).

- [ ] **Step 2: Write the failing tests**

```python
def test_list_voices(monkeypatch, tmp_path):
    for f in ["a.onnx", "a.onnx.json", "b.onnx"]: (tmp_path / f).write_bytes(b"")
    monkeypatch.setattr(app_module, "VOICES_DIR", tmp_path)
    assert app_module.list_voices() == ["a"]

def test_voices_endpoint(monkeypatch):
    monkeypatch.setattr(app_module, "list_voices", lambda: ["en_US-lessac-medium"])
    assert client.get("/api/voices").json() == {"voices": ["en_US-lessac-medium"], "default": app_module.PIPER_VOICE}

def test_speak_ok(monkeypatch):
    monkeypatch.setattr(app_module, "list_voices", lambda: [app_module.PIPER_VOICE])
    monkeypatch.setattr(app_module, "synthesize", lambda text, voice: b"RIFFfake")
    r = client.post("/api/speak", json={"text": "Hello."})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content == b"RIFFfake"

def test_speak_rejects_unknown_voice(monkeypatch):
    monkeypatch.setattr(app_module, "list_voices", lambda: ["en_US-lessac-medium"])
    r = client.post("/api/speak", json={"text": "Hi", "voice": "../../etc/passwd"})
    assert r.status_code == 400 and r.json() == {"detail": "Unknown voice."}

def test_speak_no_voices(monkeypatch):
    monkeypatch.setattr(app_module, "list_voices", lambda: [])
    assert client.post("/api/speak", json={"text": "Hi"}).status_code == 503

def test_speak_text_bounds():
    assert client.post("/api/speak", json={"text": " "}).status_code == 422
    assert client.post("/api/speak", json={"text": "a" * 1001}).status_code == 422
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_app.py -k "voice or speak" -v`
Expected: FAIL (`AttributeError` / 404)

- [ ] **Step 4: Implement `VOICES_DIR`, `list_voices`, `get_voice`, `synthesize`, `GET /api/voices` and `POST /api/speak` in `app.py`**

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: 24 passed

- [ ] **Step 6: Real round trip: Piper speaks, Whisper hears**

```bash
.venv/bin/python app.py &   # wait for "Uvicorn running on http://127.0.0.1:8765"
curl -s -X POST localhost:8765/api/speak -H 'Content-Type: application/json' \
  -d '{"text":"Privileged accounts require multi-factor authentication."}' -o /tmp/rt.wav && file /tmp/rt.wav
time curl -s -X POST localhost:8765/api/transcribe -H 'Content-Type: audio/wav' --data-binary @/tmp/rt.wav
kill %1
```
Expected: `RIFF (little-endian) data, WAVE audio`, and the transcript JSON contains `privileged accounts` and `multi-factor authentication` (case-insensitive). Note the `time` result: the first call includes the model download and load.

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_app.py && git commit -m "feat: Piper /api/speak and /api/voices"
```

---

### Task 6: `speech.js`: sentence splitting and speech cleanup

**Files:**
- Create: `static/speech.js`, `tests/speech.test.mjs`

**Interfaces:**
- Produces (ES module exports, used by `app.js` in Task 7):
  - `export function cleanForSpeech(text: string): string`: removes `*`, `_`, `` ` ``, `#`, and collapses whitespace to single spaces, trimmed.
  - `export class SentenceBuffer { push(delta: string): string[]; flush(): string[] }`: accumulates streamed text. `push` returns the sentences completed so far, and `flush` returns the remainder, or `[]` if it's only whitespace. A sentence ends at `.`, `!` or `?` followed by whitespace. It doesn't end after the abbreviations `e.g.`, `i.e.`, `etc.`, `vs.`, `No.`, `Mr.`, `Ms.`, `Dr.` (case-insensitive), and it never ends at a `.` between digits (`1.0`) because no whitespace follows it. Returned sentences are trimmed.

- [ ] **Step 1: Write the failing tests**

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { SentenceBuffer, cleanForSpeech } from '../static/speech.js';

test('emits sentences as they complete', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('ECC 2-2-3 requires MFA. It app'), ['ECC 2-2-3 requires MFA.']);
  assert.deepEqual(b.push('lies to admins! Done'), ['It applies to admins!']);
  assert.deepEqual(b.flush(), ['Done']);
});

test('does not split decimals or abbreviations', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('Version 1.0 of the policy, e.g. the access policy, applies. Next'),
    ['Version 1.0 of the policy, e.g. the access policy, applies.']);
});

test('flush of whitespace is empty', () => {
  const b = new SentenceBuffer();
  b.push('One. ');
  assert.deepEqual(b.flush(), []);
});

test('cleanForSpeech strips markdown', () => {
  assert.equal(cleanForSpeech('**MFA** is `required`\n# now'), 'MFA is required now');
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/`
Expected: FAIL with `Cannot find module .../static/speech.js`

- [ ] **Step 3: Implement `static/speech.js`**

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/`
Expected: `# pass 4`, `# fail 0`

- [ ] **Step 5: Commit**

```bash
git add static/speech.js tests/speech.test.mjs && git commit -m "feat: sentence buffer and speech cleanup for TTS"
```

---

### Task 7: The page: HIG UI, mic, streaming answer, audio queue, settings

**Files:**
- Modify: `static/index.html` (replace the placeholder)
- Create: `static/app.css`, `static/app.js`

**Interfaces:**
- Consumes: `GET /api/health`, `POST /api/transcribe`, `POST /api/ask` (SSE events `sources`/`delta`/`done`/`error`), `POST /api/speak`, `GET /api/voices` (Tasks 1–5), plus `SentenceBuffer` and `cleanForSpeech` (Task 6).
- Produces: the finished page. The exact markup is the builder's call, but these elements are required:
  - Toolbar: title "CISO Assistant", status dot plus label (`Connected` / `Knowledge base offline` / `No DeepSeek key`), and a ⚙ button that opens a `<dialog>` settings sheet.
  - `<main>` message list with `aria-live="polite"`. User bubbles go right in systemBlue with white text. Assistant bubbles go left on secondary grouped background. Under each answer, sources render as `<details>` rows showing `📄 {file_name} › {header_path}`, which expand to the excerpt and a classification badge. Errors and "Didn't catch that" render as centered secondary-label system messages.
  - Bottom bar: text input ("Type a question…", Enter sends) and a round mic button (min 44×44 px) with `aria-label` reflecting its state.

**Behaviour (app.js):**
- States: `idle → listening → transcribing → thinking → speaking → idle`. The mic button shows each state: listening has a red fill and a live level ring driven by an `AnalyserNode` RMS value; transcribing and thinking show a spinner; speaking shows a stop icon.
- **Mic:** tap to start `MediaRecorder` (default mime), tap again to stop, and the blob goes to `/api/transcribe`. Holding Space does push-to-talk, ignored while the text input has focus. If `text === ""`, show "Didn't catch that" and return to idle. Mic permission denied → system message "Microphone access was denied. Allow it in the browser's site settings."
- **Ask:** `fetch('/api/ask', {method:'POST', signal})` with `{question, history, classification}`. Parse SSE from `response.body` via a `TextDecoder`, splitting on `\n\n` (EventSource can't POST). On `delta`, append the text to the bubble and push it into `SentenceBuffer`. Each completed sentence → `cleanForSpeech` → enqueue. On `done`, `flush()` and enqueue the rest, then append `{user, assistant}` to `history`, keeping the last 6 messages. On `error`, show the message as a system message and keep any partial text.
- **Audio queue:** fetch `/api/speak` in order (prefetch the next while one plays), play via an `Audio` element with an object URL, and revoke the URL when done. Log `console.info('first audio after', ms)` measured from send to the first `play`. This is the Task 8 latency measurement.
- **Stop:** tapping the mic or pressing Esc while thinking or speaking aborts the fetch, clears the queue and pauses audio. The partial answer stays.
- **Settings sheet:** classification (`All levels` = `all`, `Exclude RESTRICTED and CONFIDENTIAL` = `exclude_sensitive`) and a voice `<select>` from `/api/voices`. Persist both in `localStorage`, wrapped in try/catch so a blocked storage falls back to defaults.
- **Health:** poll `/api/health` on load and every 15 s.

**HIG styling (app.css)**, from `~/.config/opencode/knowledge/apple-hig` (colors, typography, materials, accessibility pages):
- Font stack: `-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", system-ui, sans-serif`. Body 17px/1.4, footnote 13px for sources.
- Tokens on `:root` with dark overrides under `@media (prefers-color-scheme: dark)`: `--blue #007AFF/#0A84FF`, `--red #FF3B30/#FF453A`, `--green #34C759/#30D158`, `--bg #FFFFFF/#000000`, `--bg-grouped #F2F2F7/#1C1C1E`, `--bubble #E9E9EB/#26252A`, `--label #000/#FFF`, `--label-2 rgba(60,60,67,.6)/rgba(235,235,245,.6)`, `--separator rgba(60,60,67,.29)/rgba(84,84,88,.65)`.
- Toolbar and bottom bar: translucent material (`background: color-mix(in srgb, var(--bg) 72%, transparent); backdrop-filter: saturate(180%) blur(20px)`) with a hairline separator.
- Bubbles: 18px radius, 8×12 px padding, max-width 72%. Visible `:focus-visible` ring in `--blue`. Under `prefers-reduced-motion`, no pulse or spin animations.

- [ ] **Step 1: Build `index.html`, `app.css` and `app.js` as specified**

- [ ] **Step 2: Verify in the browser**

Start `.venv/bin/python app.py` with the tunnel up and `DEEPSEEK_API_KEY` set in `.env` (ask the user to add it; never ask them to paste it in chat). Open `http://127.0.0.1:8765` in Chrome (the chrome-devtools MCP is available) and check:
- No console errors on load. The status dot reads `Connected`.
- Typed question "What does the access control policy say about MFA?" → the answer streams in, it's spoken sentence by sentence, sources appear and expand, and `first audio after` is logged.
- Mic question gets the same result. Tapping the mic mid-answer stops audio and leaves the partial text.
- An accidental short mic tap shows "Didn't catch that".
- Settings: switch classification and voice, reload, and both are kept.
- Emulate dark mode and take screenshots of both themes. Tab through all controls; every one has a visible focus ring.
- Stop the tunnel (or set `CISO_RAG_URL` to a closed port) → the dot reads `Knowledge base offline`, and asking shows the tunnel message.

- [ ] **Step 3: Run all tests**

Run: `.venv/bin/pytest -q && node --test tests/`
Expected: 24 passed; `# pass 4`

- [ ] **Step 4: Commit**

```bash
git add static/ && git commit -m "feat: HIG voice chat page with mic, streaming answer and spoken replies"
```

---

### Task 8: README and end-to-end latency check

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

Sections: what it is (one paragraph, and what leaves the machine: only the DeepSeek request); prerequisites (Python 3.13, ffmpeg, ciso-rag SSH tunnel on `127.0.0.1:18760`, DeepSeek key); setup (`python3 -m venv .venv`, `pip install -r requirements.txt`, `cp .env.example .env` and add the key, the voice download command from Task 5 Step 1); run (`.venv/bin/python app.py`, open `http://127.0.0.1:8765`); controls (tap mic / hold Space / Esc to stop); tests (`.venv/bin/pytest`, `node --test tests/`); troubleshooting (each error message from Task 3 with its fix).

- [ ] **Step 2: Measure latency**

With the app running and the model warm, ask three spoken questions and read the `first audio after` console values.
Expected: each between roughly 3000 and 6000 ms. Record the three numbers in the README under "Performance". If any is well over 6 s, report it to the user rather than changing the design.

- [ ] **Step 3: Commit**

```bash
git add README.md && git commit -m "docs: README with setup, usage and measured latency"
```
