# Whisper Voice Assistant — Design

Date: 2026-11-15
Status: draft, awaiting approval

## Intent

A local web app you talk to. You ask a question out loud in English. The app
answers out loud from the CISO knowledge base (the `ciso-rag` server already
used by opencode) and shows the conversation as chat bubbles. The UI follows
Apple's Human Interface Guidelines (HIG).

Why: to ask the knowledge base (NCA ECC, SAMA CSF, PDPL, internal policies,
architecture docs, project plans, Apple HIG) questions by voice and hear a
short, grounded answer, without opening opencode.

## Goals

- Speak a question, hear a 2–4 sentence answer grounded in ciso-rag passages.
- See the question, the answer (text appears while it is spoken) and the
  source documents under each answer.
- Typing also works.
- Feels like a native macOS app (Messages-like), light and dark mode.
- Starts with one command: `python app.py`, then open `http://127.0.0.1:<port>`.
- Latency: about 3–6 s from tapping "send" to hearing the first word is
  acceptable (user confirmed). Whole-clip upload, no streaming transcription
  or voice-activity detection. Measure it once built; speeding up later is a
  contained change.

## Non-goals (skipped on purpose)

- Arabic or any language other than English.
- Saved chat history (reload clears it), accounts, multi-user.
- Wake word or always-listening mode.
- Query rewriting for follow-ups (search uses only the latest question;
  DeepSeek sees recent turns). Known limit.
- Triggering `ciso_rag_sync` or listing documents from the UI.

## Anti-goals (would make it a failure even if it works)

- Answers that invent content not in the retrieved passages.
- The server reachable from other machines (it must bind to 127.0.0.1 only).
- The DeepSeek API key ending up in the page, logs, or the repo.

## Constraints (verified)

- Machine: Linux, 12-core CPU, 16 GB RAM, **no GPU**. Python 3.13, Node 24,
  ffmpeg installed.
- ciso-rag: HTTP API at `http://127.0.0.1:18760` (an SSH tunnel; must be up).
  - `GET /api/health` → `{status, engine, collection, indexed_chunks, indexed_documents}`
  - `POST /api/v1/search` body `{query, limit, rerank: true, domain?, framework?, classification?}`
    → `{query, total_results, results: [{score, text, header_path, file_name, domain, framework, classification, chunk_index, ...}]}`
  - `classification` filter is a single exact value, not a "max level".
  - No auth header or token: the MCP wrapper sends only `Accept` /
    `Content-Type`, and a sample search with no auth succeeded.
  - Classification labels (uppercase, exact): `PUBLIC`, `INTERNAL`,
    `RESTRICTED`, `CONFIDENTIAL` (from the wrapper's enum; sample result
    returned `"INTERNAL"`).
  - Same API the opencode MCP wrapper (`~/.config/opencode/openclaw/runtime/ciso-rag-mcp.mjs`) calls.
- DeepSeek: OpenAI-compatible API, base URL `https://api.deepseek.com`,
  model `deepseek-chat`, streaming supported. User has a key.
- Mic access in the browser needs a secure context; `http://127.0.0.1` qualifies.
- What leaves the machine: only the DeepSeek request (question, recent turns,
  retrieved passages, which may include RESTRICTED/CONFIDENTIAL text). User
  accepted this. Voice audio and speech synthesis stay local.

## Approaches considered

| | Approach | Trade-off |
|---|---|---|
| **A (picked)** | Python FastAPI server; faster-whisper `small.en` (CPU, int8); ciso-rag; DeepSeek streaming; Piper local TTS | Everything except DeepSeek is local; good English voice; one process |
| B | Same, with Microsoft neural voices (`edge-tts`) | Slightly more expressive voice, but every answer's text goes to Microsoft and speech fails offline |
| C | Node server + whisper.cpp | Matches the Node MCP stack, but needs compiling whisper.cpp and has weaker TTS options |
| — | Whisper in the browser (WASM) | Rejected: too slow on CPU |

Earlier, Arabic support pointed to large-v3-turbo + edge-tts; the user then
narrowed to English only, which made A (small.en + Piper) the clear pick.
User chose A.

## Design

### Flow

```
mic (browser, MediaRecorder webm/opus)
  → POST /api/transcribe          faster-whisper small.en → text
  → POST /api/ask  (SSE stream)    ciso-rag search → DeepSeek stream
        events: sources, delta…, done | error
  → browser splits deltas into sentences
  → POST /api/speak per sentence → Piper → WAV, played in order
```

### Server (`app.py`, FastAPI, uvicorn on 127.0.0.1)

- `GET /` serves the static page (`static/index.html`, `app.css`, `app.js`).
- `GET /api/health` → proxies ciso-rag health; the page shows a status dot.
- `POST /api/transcribe` → audio blob in, `{text}` out. Whisper model loaded
  once at startup. Empty/silent audio → empty text, UI says "Didn't catch that".
- `POST /api/ask` → `{question, history: [{role, content}…], classification}`.
  1. Search ciso-rag (`limit` 5, `rerank: true`).
  2. Send `sources` event (file_name, header_path, classification, text excerpt).
  3. Stream DeepSeek deltas as `delta` events, then `done`.
  4. Any failure → `error` event with a plain message.
- `POST /api/speak` → `{text}` in, WAV out (Piper, voice loaded once).
- `GET /api/voices` → Piper voice files present in `voices/`.

### Prompt to DeepSeek

System prompt (builder may tune wording):
- Answer only from the provided passages. If they don't cover it, say you
  couldn't find it in the knowledge base.
- 2–4 sentences, plain spoken English: no markdown, lists, or citations
  read aloud (sources are shown in the UI instead).
- Passages are given with their document name and section.
- History: last 3 exchanges.

### Classification setting (my call)

Two choices in settings: **All levels** (default) or **Exclude RESTRICTED and
CONFIDENTIAL**. Because ciso-rag's filter is exact-match, the second option is
done server-side: request 10 results, drop those two levels, keep the top 5.
**Fail-closed** (my call): in Exclude mode, a passage whose classification is
missing or not one of the four known labels is dropped too.

### No-results guard (my call)

If the search returns zero passages (after filtering), the server does not
call DeepSeek. It sends a fixed answer, "I couldn't find that in the knowledge
base.", as `delta` + `done`, with an empty `sources` event. No score
threshold: reranked scores aren't calibrated, so beyond zero results the
system prompt handles weak matches.

### Page (single HTML page, vanilla JS, no build step)

```
┌──────────────────────────────────────────────────────┐
│  CISO Assistant                          ● Connected  ⚙ │
├──────────────────────────────────────────────────────┤
│                    ┌───────────────────────────────┐ │
│                    │ user question (blue, right)   │ │
│                    └───────────────────────────────┘ │
│  ┌──────────────────────────────────────┐            │
│  │ answer (grey, left), streams in      │            │
│  └──────────────────────────────────────┘            │
│   📄 Document › Section                   ›  (tap: excerpt + classification)
├──────────────────────────────────────────────────────┤
│   [ Type a question…                     ]   ( 🎤 )  │
└──────────────────────────────────────────────────────┘
```

- Mic: tap to start, tap to send; hold Space for push-to-talk. Live level
  meter while listening. Tapping the mic while speaking stops playback.
- States the mic button shows: idle, listening, transcribing, thinking,
  speaking.
- Settings sheet (⚙): classification setting, Piper voice.
- Errors shown inline as a system message: ciso-rag unreachable ("Is the SSH
  tunnel up?"), DeepSeek error, mic permission denied.

### HIG styling

Reference: local snapshot `~/.config/opencode/knowledge/apple-hig` (also
searchable in ciso-rag with `domain=hig`).
- System font stack (`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", system-ui, sans-serif`).
- Apple system colors as CSS variables, light and dark (`prefers-color-scheme`);
  e.g. systemBlue `#007AFF` / `#0A84FF`, grouped/secondary backgrounds, labels.
- Translucent toolbar and input bar (backdrop blur material).
- Minimum 44×44 pt hit targets, visible focus rings, keyboard operable,
  `prefers-reduced-motion` respected, VoiceOver-friendly labels / `aria-live`
  on the answer.

### Config (`.env`, git-ignored; `.env.example` committed)

```
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-chat
CISO_RAG_URL=http://127.0.0.1:18760
WHISPER_MODEL=small.en
PIPER_VOICE=en_US-lessac-medium
PORT=8765
```

### Files

```
app.py              FastAPI server (all endpoints)
static/index.html
static/app.css
static/app.js
voices/             Piper .onnx + .onnx.json (downloaded by setup step)
requirements.txt    fastapi, uvicorn, faster-whisper, piper-tts, httpx, python-dotenv
.env.example
README.md           setup + run
tests/test_app.py   small checks (below)
```

### Checks

- Unit: classification post-filter, prompt/history building, SSE event shape
  (ciso-rag and DeepSeek mocked).
- Manual: speak a question end to end with the tunnel up; tunnel down shows
  the error; light and dark mode.

## Left to the builder

- Exact sentence-splitting rule for TTS and audio queue mechanics.
- Exact system-prompt wording.
- How Piper voice files are downloaded (script or README step).
- If `piper-tts` or `faster-whisper` has no Python 3.13 wheel, use a 3.12 venv
  (via `uv`) and say so in the README.
- Port number.
