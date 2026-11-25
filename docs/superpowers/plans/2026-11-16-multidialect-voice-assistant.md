# Multidialect Voice Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the English CISO knowledge-base assistant into "Rawi", a demo voice assistant that hears English, Moroccan, Egyptian and Saudi Arabic and answers out loud in the same language and dialect, with speech in and out running on a mini PC and replies from DeepSeek.

**Architecture:** The FastAPI app is split into `config.py`, `dialects.py`, `llm.py`, `stt.py`, `tts.py` and the routes in `app.py`. DeepSeek starts every reply with a dialect tag (`[ar-MA]`…); the server strips it, sends it to the page as a `dialect` SSE event, and the page asks `/api/speak` for that dialect's voice. Models are chosen by measurement (Tasks 4–8) before the speech modules are finished (Tasks 9–10), then the app is deployed and rehearsed on the mini PC (Tasks 12–14).

**Tech Stack:** Python 3.13, FastAPI, uvicorn, openai SDK → DeepSeek `deepseek-chat`, faster-whisper 1.2.1 (plus a Darija model only if Task 8 picks one), Piper 1.8 plus the Arabic voice engines Task 8 picks, vanilla JS ES modules, pytest, `node --test`, `arabic-asr-benchmark` (bench only).

**Spec:** `docs/superpowers/specs/2026-11-16-multidialect-voice-assistant-design.md` (schema page: https://claude.ai/artifact/E8md7NPwZX1wwZnhXLvs6d)

## Global Constraints

- Dialect codes are exactly `en`, `ar-MA`, `ar-EG`, `ar-SA`. Arabic that is none of the three (MSA, Levantine, …) is answered as `ar-SA` (`DEFAULT_ARABIC = "ar-SA"`): the clients are in Saudi Arabia.
- Persona name: "Rawi" (Arabic "راوي"). Page title and header: "Rawi".
- Replies are 2–4 short spoken sentences, no markdown, no lists.
- Reply language and dialect follow the **latest** question, not the history.
- Mixed Arabic + English question → reply in the Arabic dialect, English terms kept as English words.
- Topics: polite and short; steer away from politics and religion.
- Only text goes to DeepSeek. Audio never leaves the mini PC.
- No knowledge-base code remains: no ciso-rag, no `search_knowledge_base`, no sources, no classification.
- The DeepSeek key is never read, printed, logged, copied or sent to the page. `.env` is never read by the agent and never copied to the mini PC; the user creates it there.
- Speech-to-text and voice models must be open and downloadable, and their license must allow a client demo. The license is recorded in Task 8's decision record before a model is adopted.
- The server binds to `127.0.0.1` by default; LAN serving is opt-in via env (Task 12).
- Error copy, exact:
  - `NO_KEY = "DeepSeek API key missing. Add DEEPSEEK_API_KEY to .env and restart."`
  - `LLM_FAILED = "DeepSeek request failed. Try again."`
  - `OFFLINE = {"en": "I can't reach my reply service right now. Please check the internet connection.", "ar": "ما أقدر أوصل لخدمة الردود الحين. تأكد من اتصال الإنترنت."}`
- Python deps keep the existing pins (`faster-whisper==1.2.1`, `av>=11,<17`, `piper-tts==1.8.0`). pip installs use `TMPDIR=~/.cache/pip-tmp` (small `/tmp`).
- Test commands: `.venv/bin/pytest -q` and `node --test tests/*.test.mjs` (Node 24 rejects `node --test tests/`).
- Timing targets (spec): first audio about 2–3 s after the speaker stops, measured on the mini PC per dialect.

## Review Focus

1. **Microphone from another device on the LAN.** Browsers only allow the mic on `https://` or `localhost`. Opening `http://<minipc>:8765` from the demo laptop silently loses the mic. Expected: a documented access path that keeps the mic: an SSH port-forward to `localhost`, or HTTPS with a trusted local certificate. Pinned by Task 12's `test_ssl_settings_from_env` and `test_lan_host_allowed`, and by the runbook step in Task 14.
2. **DeepSeek omits, garbles or splits the dialect tag.** A tag split across chunks (`"[ar-"`, `"MA] مرحبا"`), a tag with spaces (`"[ ar-EG ]"`), an unknown tag (`"[fr]"`), or no tag at all must never be spoken or shown. The voice still follows the reply's script. Pinned by Task 3's parser tests.
3. **Whisper detects Darija as another language** (French, Persian, …) and transcribes it in the wrong script. Expected: the transcript is redone restricted to whichever of en/ar is more likely. Pinned by Task 9's `test_non_en_ar_detection_retries_with_best_of_en_ar`.
4. **Internet drops mid-answer.** Expected:
   - Text already shown and spoken stays.
   - Then a spoken and shown offline message in the question's language. Not silence, and not the generic failure message.

   Pinned by Task 3's `test_ask_offline_mid_stream` and Task 11's page handling.
5. **Arabic sentence splitting.** `؟` ends a sentence, `،` does not, and decimal numbers like `3.5` don't split. Otherwise the first Arabic sentence is never released to the voice and first audio waits for the whole reply. Pinned by Task 11's node tests.

---

## Overview of tasks

| # | Task | Kind | Depends on | User gate |
|---|---|---|---|---|
| 1 | Branch; remove the knowledge base | TDD | — | |
| 2 | Split `app.py` into modules | refactor (tests stay green) | 1 | |
| 3 | Dialect prompt, tag parser, `dialect` event, offline errors | TDD | 2 | |
| 4 | DeepSeek dialect samples | measurement | 3 | **G1: user judges the dialects** |
| 5 | Mini PC hardware check | measurement | SSH address | **G2: user confirms the tier** |
| 6 | Speech-to-text benchmark on the mini PC | measurement | 5 | |
| 7 | Voice samples on the mini PC | measurement | 5 | **G3: user judges the voices** |
| 8 | Model decision record | doc | 4, 6, 7 | **G4: user approves the models** |
| 9 | Multilingual speech-to-text (`stt.py`) | TDD | 2, 8 | |
| 10 | Voice per dialect (`tts.py`), `/api/speak` by dialect, health | TDD | 2, 8 | |
| 11 | Page: RTL, badge, Arabic splitting, voice per dialect, offline message | TDD (node) + browser check | 3, 10 | |
| 12 | LAN access, HTTPS option, stage timing logs | TDD | 2 | |
| 13 | Deploy to the mini PC | ops | 9–12 | |
| 14 | Rehearsal harness, measurements, README and runbook | measurement + docs | 13 | **G5: user runs the demo script** |

Tasks 1–3 and 12 need neither the mini PC nor any gate, so they can run while waiting for the SSH address. Tasks 4–8 are **measurement tasks**: they produce reports for the user, have no failing-test step, and stop at their gate until the user answers.

---

### Task 1: Branch and remove the knowledge base

**Files:**
- Modify: `app.py` (remove lines 39–72 ciso-rag, 87–113 `SEARCH_TOOL`/`NO_PASSAGES`/`SEARCH_LIMIT`/`format_passages`, `RAG_DOWN`, `ToolCall`, tool handling in `stream_turn`, `MAX_SEARCHES` loop in `ask_events`, `classification` in `AskBody`, rag in `/api/health`)
- Modify: `static/app.js` (`renderSources`, sources container, `classification` in the ask body and `state.settings`, the `localStorage.removeItem('whisper.classification')` line, health handling of `rag`)
- Modify: `static/app.css` (remove `.sources`, `.source`, `.excerpt`, `.badge[data-level]` rules)
- Modify: `.env.example` (remove `CISO_RAG_URL`)
- Modify: `tests/test_app.py`

**Interfaces:**
- Produces:
  - `stream_turn(messages: list[dict]) -> AsyncIterator[str]` (text chunks only; no `tools` parameter).
  - `ask_events(body: AskBody) -> AsyncIterator[str]` yields `delta`, `done` and `error` only.
  - `GET /api/health` returns `{"deepseek_key": bool}`.
  - `AskBody = {question: str (1..2000), history: list[Turn]}`.

- [ ] **Step 1: Create the branch**

```bash
git checkout -b feat/multidialect-voice-assistant
```

- [ ] **Step 2: Rewrite the tests for the plain chat flow (failing)**

In `tests/test_app.py`:
- Delete these tests:
  - `test_health_ok`, `test_health_rag_down`, `test_health_rag_not_json`
  - `test_filter_*`, `test_search_rag_payload`, `test_format_passages`
  - every `test_stream_turn_*` that uses `tc(...)` or `SEARCH_TOOL`
  - `test_ask_document_question_searches`, `test_ask_empty_query_uses_question`, `test_ask_empty_search_tells_model`
  - `test_ask_search_cap_*`, `test_ask_rag_*`, `test_ask_small_talk_works_with_tunnel_down`, `test_ask_separates_text_across_turns`
- Delete these helpers: `P`, `fake_search`, `TC` and `tc`.

Then add:

```python
def test_health_reports_key(monkeypatch):
    monkeypatch.setattr(app_module, "DEEPSEEK_API_KEY", "sk-test")
    assert client.get("/api/health").json() == {"deepseek_key": True}
    monkeypatch.setattr(app_module, "DEEPSEEK_API_KEY", "")
    assert client.get("/api/health").json() == {"deepseek_key": False}

def test_stream_turn_sends_no_tools(monkeypatch):
    seen = fake_deepseek(monkeypatch, [text("Hello.")])
    assert collect([{"role": "user", "content": "q"}]) == ["Hello."]
    assert "tools" not in seen["kw"]

def test_ask_streams_reply(monkeypatch, keyed):
    calls = fake_turns(monkeypatch, ["Hi! ", "How can I help?"])
    assert events(ask(question="hello")) == [
        ("delta", {"text": "Hi! "}), ("delta", {"text": "How can I help?"}), ("done", {})]
    assert calls[0]["messages"][-1] == {"role": "user", "content": "hello"}

def test_ask_ignores_old_classification_field(monkeypatch, keyed):
    fake_turns(monkeypatch, ["Hi."])
    assert ask(question="hello", classification="all").status_code == 200

def test_no_knowledge_base_left():
    for name in ("search_rag", "rag_get", "SEARCH_TOOL", "ToolCall", "CISO_RAG_URL", "RAG_DOWN"):
        assert not hasattr(app_module, name), name
```

Update the helpers: `collect(messages)` takes no `tools`, and `fake_turns`'s inner `turn(messages)` takes no `tools`. Keep the existing tests for history passing, LLM failure mid-stream, no key, validation and client wiring, updated for the new signatures.

- [ ] **Step 3: Run the tests and watch them fail**

Run: `.venv/bin/pytest -q`
Expected: FAIL. `test_health_reports_key` gets extra keys, `test_no_knowledge_base_left` names `search_rag`, and the stream tests hit the old signatures.

- [ ] **Step 4: Remove the knowledge-base code from `app.py`**

`ask_events` becomes one DeepSeek turn: key check → `build_messages` → forward each chunk as `delta` → `done`. On an exception it logs the type name only and sends `error` `LLM_FAILED`. `SYSTEM_PROMPT` becomes a placeholder, `"You are a friendly voice assistant. Reply in 2 to 4 short spoken sentences, no markdown."`, which Task 3 replaces.

- [ ] **Step 5: Remove sources and classification from the page**

In `app.js`:
- Drop `renderSources`, the `sources` element in `ask()`, the `sources` event branch, `classification` from `state.settings` and from the request body, and the `localStorage.removeItem` line.
- In `checkHealth`, show "No DeepSeek key" when `!h.deepseek_key`, else "Connected". Keep "Server unreachable" on fetch failure.
- Set the empty hint in `index.html` to "Talk to me in English or Arabic." Its final wording comes in Task 11.

In `app.css`, delete the source and badge rules listed under Files.

- [ ] **Step 6: Run both suites**

Run: `.venv/bin/pytest -q && node --test tests/*.test.mjs`
Expected: all pass. Node: 9 pass.

- [ ] **Step 7: Check in the browser**

Restart the app (`.venv/bin/python app.py`) and ask "hi" by text. Expected:
- A streamed reply that is spoken.
- No sources area.
- No console errors.
- `grep -rn "rag\|classification\|sources" static/` returns nothing.

- [ ] **Step 8: Commit**

```bash
git add -A app.py static tests .env.example
git commit -m "refactor: remove the knowledge base; plain DeepSeek conversation"
```

---

### Task 2: Split `app.py` into modules

A pure refactor: behaviour and test count stay the same, and only import paths change.

**Files:**
- Create: `config.py`, `dialects.py` (empty constants for now, filled in Task 3), `llm.py`, `stt.py`, `tts.py`
- Modify: `app.py` (routes, middleware, lifespan, `__main__` only)
- Create: `tests/test_llm.py`, `tests/test_stt.py`, `tests/test_tts.py` (move the matching tests out of `tests/test_app.py`)
- Create: `tests/conftest.py` (shared `client`, `events`, `keyed`, `fake_deepseek`, `fake_turns`)

**Interfaces:**
- `config.py` holds `BASE_DIR`, `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `WHISPER_MODEL`, `PIPER_VOICE`, `PORT`, all read from env after `load_dotenv(BASE_DIR / ".env")`.
- `llm.py` holds `SYSTEM_PROMPT`, `build_messages(question: str, history: list[dict]) -> list[dict]`, `get_deepseek()`, `close_deepseek()`, `stream_turn(messages: list[dict]) -> AsyncIterator[str]`, `NO_KEY` and `LLM_FAILED`.
- `stt.py` holds `get_whisper()`, `transcribe_audio(data: bytes) -> str` and `MAX_AUDIO_BYTES`.
- `tts.py` holds `VOICES_DIR`, `list_voices()`, `get_voice(name)` and `synthesize(text, voice) -> bytes`.
- `app.py` holds `app`, `SameOriginMiddleware`, `sse(event, data) -> str`, `AskBody`, `Turn`, `SpeakBody`, `ask_events`, the routes and `preload_models()`.
- **Rule for every later task:** `app.py` calls the other modules through the module object (`import llm` … `llm.stream_turn(...)`), never `from llm import stream_turn`. That way tests can `monkeypatch.setattr(llm, "stream_turn", fake)`. Inside a module, anything a test fakes is called through the module global. For example, `stt.transcribe_audio` calls `get_whisper()`.
- Tests that set the key patch `config.DEEPSEEK_API_KEY`, and every reader looks it up as `config.DEEPSEEK_API_KEY` at call time.

- [ ] **Step 1: Move the shared test helpers into `tests/conftest.py`.** Fixtures: `client` and `keyed`. Plain helpers: `events(r)`, `fake_deepseek(monkeypatch, chunks)` (patches `llm.get_deepseek`), `fake_turns(monkeypatch, *turns)` (patches `llm.stream_turn`), `text(c)` and `collect(messages)`. Set `pythonpath = . tests` in `pytest.ini` so test files can `from conftest import ...`.
- [ ] **Step 2: Move the tests** to `tests/test_llm.py` (stream/client wiring/build_messages), `tests/test_stt.py` (transcribe and Whisper settings) and `tests/test_tts.py` (voices and speak). Repoint each `monkeypatch.setattr(app_module, X)` at the module that now owns `X`.
- [ ] **Step 3: Run the tests and watch them fail.** Run `.venv/bin/pytest -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'llm'` (and `stt`, `tts`).
- [ ] **Step 4: Create the modules** by moving code from `app.py` unchanged, and update `app.py` to call through module objects.
- [ ] **Step 5: Run both suites.** Expected: same count as at the end of Task 1, all pass.
- [ ] **Step 6: Smoke run.** Run `.venv/bin/python app.py`, then type "hi". Expected: a spoken reply.
- [ ] **Step 7: Commit**

```bash
git add -A config.py dialects.py llm.py stt.py tts.py app.py tests pytest.ini
git commit -m "refactor: split app.py into config, llm, stt and tts modules"
```

---

### Task 3: Dialect prompt, tag parser, `dialect` event, offline errors

**Files:**
- Modify: `dialects.py`, `llm.py`, `app.py`
- Test: `tests/test_dialects.py`, `tests/test_llm.py`, `tests/test_app.py`

**Interfaces:**
- Produces (`dialects.py`):
  - `DIALECTS: tuple[str, ...] = ("en", "ar-MA", "ar-EG", "ar-SA")` and `DEFAULT_ARABIC = "ar-SA"`.
  - `Dialect = Literal["en", "ar-MA", "ar-EG", "ar-SA"]`.
  - `has_arabic(text: str) -> bool`: true when any char is in U+0600–U+06FF or U+0750–U+077F.
  - `language_of(dialect: str) -> Literal["en", "ar"]`.
  - `guess_dialect(text: str) -> Dialect`: `DEFAULT_ARABIC` when `has_arabic(text)`, else `"en"`.
- Produces (`llm.py`):
  - `SYSTEM_PROMPT: str` (exact text below).
  - `class DialectTagParser` with `feed(chunk: str) -> tuple[Dialect | None, str]` and `finish() -> tuple[Dialect | None, str]`. The returned dialect is non-`None` exactly once: on the call that resolves the tag.
  - `OFFLINE: dict[str, str]` (copy in Global Constraints) and `OfflineError(Exception)`.
  - `stream_turn` raises `OfflineError` when the cause is `openai.APIConnectionError`. `openai.APITimeoutError` is a subclass of it, so timeouts are covered too. Other exceptions propagate unchanged.
- Produces (`app.py`):
  - The SSE contract:
    - `dialect` `{"dialect": Dialect}`, exactly once, before the first `delta`;
    - `delta` `{"text": str}`;
    - `done` `{}`;
    - `error` `{"message": str, "kind": "nokey" | "offline" | "failed", "language": "en" | "ar"}`.
  - `AskBody.language: Literal["en", "ar"] | None = None`, the question's language from STT or the page. It picks the language of the `offline` message. When it's `None`, the server uses `language_of(guess_dialect(question))`.

- [ ] **Step 1: Write the failing tests for `dialects.py`**

```python
import dialects as d

def test_dialect_codes():
    assert d.DIALECTS == ("en", "ar-MA", "ar-EG", "ar-SA") and d.DEFAULT_ARABIC == "ar-SA"

def test_has_arabic():
    assert d.has_arabic("وش أخبارك") and d.has_arabic("meeting بكرة") and not d.has_arabic("Hello 3.5!")

def test_guess_and_language():
    assert d.guess_dialect("كيداير") == "ar-SA" and d.guess_dialect("hi") == "en"
    assert d.language_of("ar-MA") == "ar" and d.language_of("en") == "en"
```

- [ ] **Step 2: Write the failing parser tests in `tests/test_llm.py`**

```python
from llm import DialectTagParser as TP

def run(chunks):
    p, out, tags = TP(), [], []
    for c in chunks:
        tag, t = p.feed(c); tags += [tag] if tag else []; out.append(t)
    tag, t = p.finish(); tags += [tag] if tag else []; out.append(t)
    return tags, "".join(out)

def test_tag_in_one_chunk():
    assert run(["[ar-MA] لاباس، الحمد لله."]) == (["ar-MA"], "لاباس، الحمد لله.")

def test_tag_split_across_chunks():
    assert run(["[ar-", "EG]", " إزيك؟"]) == (["ar-EG"], "إزيك؟")

def test_tag_with_spaces_and_case():
    assert run(["[ AR-sa ] هلا"]) == (["ar-SA"], "هلا")

def test_unknown_tag_dropped_and_guessed_from_text():
    assert run(["[fr] Bonjour, how are you?"]) == (["en"], "Bonjour, how are you?")

def test_missing_tag_guesses_from_script():
    assert run(["هلا والله، ", "كيف حالك؟"]) == (["ar-SA"], "هلا والله، كيف حالك؟")
    assert run(["Hello there."]) == (["en"], "Hello there.")

def test_bracket_later_in_text_is_kept():
    assert run(["[en] Use [brackets] freely."]) == (["en"], "Use [brackets] freely.")

def test_stream_shorter_than_lookahead_resolves_on_finish():
    assert run(["Hi"]) == (["en"], "Hi")

def test_empty_stream():
    assert run([]) == (["en"], "")
```

The parser algorithm isn't fixed by the signature, so here it is:

```
buffer chunks until resolved:
  s = buffer.lstrip()
  if s starts with "[":
      if "]" in s[:16]: tag = s[1:s.index("]")].strip(); text = s after "]" with leading whitespace removed
                        dialect = canonical(tag) if it matches one of DIALECTS case-insensitively, else guess_dialect(text)
      elif len(s) >= 16: no tag; dialect = guess_dialect(s); text = s
      else: keep buffering
  elif s has at least 8 non-space chars: no tag; dialect = guess_dialect(s); text = s
  else: keep buffering (so "Hi" + "، هلا" is not guessed from "Hi" alone)
after resolution: feed returns (None, chunk) unchanged
finish(): resolve whatever is buffered with the same rules, without the length waits (empty → ("en", ""))
```

- [ ] **Step 3: Write the failing prompt and stream tests in `tests/test_llm.py`**

```python
import llm, openai, httpx

def test_system_prompt_rules():
    p = llm.SYSTEM_PROMPT
    for must in ("[en]", "[ar-MA]", "[ar-EG]", "[ar-SA]", "Rawi", "latest", "politics", "religion", "2 to 4"):
        assert must in p, must

def test_offline_is_raised_for_connection_errors(monkeypatch):
    async def create(**kw): raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com"))
    monkeypatch.setattr(llm, "get_deepseek", lambda: NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(llm.OfflineError):
        collect([{"role": "user", "content": "q"}])
```

- [ ] **Step 4: Write the failing route tests in `tests/test_app.py`**

```python
def test_ask_sends_dialect_then_text(monkeypatch, keyed):
    fake_turns(monkeypatch, ["[ar-", "MA] لاباس", "، الحمد لله."])
    assert events(ask(question="لاباس عليك؟")) == [
        ("dialect", {"dialect": "ar-MA"}), ("delta", {"text": "لاباس"}),
        ("delta", {"text": "، الحمد لله."}), ("done", {})]

def test_ask_no_tag_still_sends_dialect(monkeypatch, keyed):
    fake_turns(monkeypatch, ["Hello there, ", "how can I help?"])
    ev = events(ask(question="hi"))
    assert ev[0] == ("dialect", {"dialect": "en"}) and ev[-1] == ("done", {})

def test_ask_offline_before_any_text(monkeypatch, keyed):
    fake_turns(monkeypatch, [llm.OfflineError()])
    assert events(ask(question="وش أخبارك؟")) == [
        ("error", {"message": llm.OFFLINE["ar"], "kind": "offline", "language": "ar"})]

def test_ask_offline_mid_stream(monkeypatch, keyed):
    fake_turns(monkeypatch, ["[en] Sure, ", "here is", llm.OfflineError()])
    ev = events(ask(question="hi", language="en"))
    assert ev[:3] == [("dialect", {"dialect": "en"}), ("delta", {"text": "Sure, "}), ("delta", {"text": "here is"})]
    assert ev[3] == ("error", {"message": llm.OFFLINE["en"], "kind": "offline", "language": "en"})

def test_ask_failure_kind(monkeypatch, keyed):
    fake_turns(monkeypatch, [RuntimeError("500")])
    assert events(ask(question="hi"))[-1] == ("error", {"message": llm.LLM_FAILED, "kind": "failed", "language": "en"})

def test_ask_no_key_kind(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    assert events(ask(question="hi")) == [("error", {"message": llm.NO_KEY, "kind": "nokey", "language": "en"})]

def test_ask_language_validated():
    assert client.post("/api/ask", json={"question": "q", "language": "fr"}).status_code == 422
```

`fake_turns` already raises an `Exception` instance found in a turn's list at that point; keep that behaviour.

- [ ] **Step 5: Run the tests and watch them fail.** Run `.venv/bin/pytest -q`. Expected: FAIL on the missing `DialectTagParser`, `OFFLINE`, `OfflineError` and `dialects` names, the prompt assertions, and the missing `dialect` event and `kind`.

- [ ] **Step 6: Implement `dialects.py`, the parser, `OfflineError` and `OFFLINE` in `llm.py`, and the new `ask_events`**

`ask_events` order:
1. Work out `lang = body.language or language_of(guess_dialect(body.question))`.
2. If there's no key, send `error` `nokey`.
3. Otherwise stream through `DialectTagParser`.
4. Send `dialect` on the resolving call, before that call's text.
5. Send `delta` only for non-empty text.
6. On `finish()`, send any `dialect` that hasn't gone out yet and the remaining text.
7. Send `done`.

`OfflineError` maps to kind `offline` with `OFFLINE[lang]`. Any other exception maps to `failed` with `LLM_FAILED`. Log only the exception type name, as today. Add `DEEPSEEK_BASE_URL` to `config.py` (default `https://api.deepseek.com`) and use it in `get_deepseek()`. Task 11 uses it to simulate an outage.

`SYSTEM_PROMPT`, exact:

```
You are Rawi (راوي), a friendly voice assistant at a live demo. People talk to you in English or in Moroccan, Egyptian or Saudi Arabic, and you hear them through speech recognition, so forgive small transcription errors.
Start every reply with exactly one tag naming the language of your reply: [en], [ar-MA], [ar-EG] or [ar-SA]. Then a space, then the reply.
Always reply in the language and dialect of the latest message, even if earlier messages used another one: English → [en]; Moroccan Darija → [ar-MA] in Darija; Egyptian → [ar-EG] in Egyptian; Saudi or any other Arabic → [ar-SA] in Saudi dialect.
If the latest message mixes Arabic and English, reply in its Arabic dialect and keep the English terms as English words.
Write Arabic in Arabic script, the way people of that dialect write casually, never transliterated.
Reply in 2 to 4 short spoken sentences. No markdown, lists, emojis or headings.
Be warm and brief. Do not discuss politics or religion: say politely that you're a demo assistant and offer another topic.
```

- [ ] **Step 7: Run both suites.** Expected: all pass.
- [ ] **Step 8: Commit**

```bash
git add config.py dialects.py llm.py app.py tests
git commit -m "feat: dialect-tagged replies, dialect SSE event, offline errors"
```

---

### Task 4: DeepSeek dialect samples (measurement → gate G1)

**Files:**
- Create: `bench/llm_dialects.py`, `bench/questions.json`
- Modify: `.gitignore` (add `bench/results/`)
- Output: `bench/results/llm_dialects.md`

**Interfaces:**
- Consumes: `llm.build_messages`, `llm.stream_turn`, `llm.DialectTagParser`, `dialects.DIALECTS`.

- [ ] **Step 1: Write `bench/questions.json`** with exactly these sets:

```json
{
  "en":    ["Hi, how are you today?", "What's a good way to start the morning?", "Can you suggest a quick dinner idea?", "Why is the sky blue?", "What can you do?"],
  "ar-MA": ["لاباس عليك؟ كيداير اليوم؟", "شنو هو أحسن وقت باش نزور مراكش؟", "عطيني فكرة على شي ماكلة مغربية ساهلة", "علاش السما زرقا؟", "شنو كتعرف دير؟"],
  "ar-EG": ["إزيك؟ عامل إيه النهارده؟", "إيه أحسن مكان أزوره في القاهرة؟", "قولي أكلة مصرية سهلة أعملها", "ليه السما لونها أزرق؟", "إنت بتعرف تعمل إيه؟"],
  "ar-SA": ["وش أخبارك اليوم؟", "وش أحسن مكان أزوره في الرياض؟", "عطني فكرة أكلة سعودية سهلة", "ليش السما لونها أزرق؟", "وش تقدر تسوي؟"],
  "mixed": ["عندي meeting بكرة، وش الـ plan؟", "كيفاش نوجد presentation مزيانة؟", "عايز أعمل backup للموبايل، أعمل إيه؟"],
  "switch": ["Hi, who are you?", "لاباس عليك؟", "طب إنت منين؟", "وش تنصحني أسوي اليوم؟", "Thanks, bye!"],
  "guard": ["What do you think about the elections?", "وش رأيك في الدين؟"]
}
```

Expected dialects:
- `mixed`: `ar-SA`, `ar-MA`, `ar-EG`, in that order.
- `switch`: `en`, `ar-MA`, `ar-EG`, `ar-SA`, `en`.
- `guard`: `en`, `ar-SA`.

- [ ] **Step 2: Write `bench/llm_dialects.py`.** Run each question as a fresh conversation, except `switch`, which runs as one conversation passing history between turns. For each question, record:
  - the expected dialect;
  - the tag DeepSeek returned;
  - the reply text;
  - time to the first text chunk, in ms.

  Write a Markdown table per set to `bench/results/llm_dialects.md`, plus a summary line: tags matching the expected dialect, out of the total. The script reads the key only through `config` (env/.env) and never prints it.

- [ ] **Step 3: Run it.** Run `.venv/bin/python bench/llm_dialects.py`. Expected: 30 rows, and a tag match of at least 27/30. For each mismatch, read the reply and note it.

- [ ] **Step 4: Gate G1.** Show the user the replies grouped by dialect and ask them to mark each dialect natural / acceptable / wrong.
  - **If a dialect is wrong:** add 2 short example exchanges for it to `SYSTEM_PROMPT` (few-shot), rerun, and show the user again. Record the outcome in the ledger as a ruling.
  - **If Moroccan is still wrong after that:** record it for Task 8. The fallback is Atlas-Chat, which brings back a local language model for Darija only. Raise it with the user in Task 8; don't decide it alone.

- [ ] **Step 5: Commit the script and questions** (not the results).

```bash
git add bench/llm_dialects.py bench/questions.json .gitignore && git commit -m "bench: DeepSeek dialect samples"
```

---

### Task 5: Mini PC hardware check (measurement → gate G2)

**Needs:** the SSH address from the user (`user@host`). If it's missing, stop and ask; this is the plan's first hard dependency on the user.

**Files:**
- Create: `bench/hardware.sh`
- Output: `bench/results/hardware.txt`

- [ ] **Step 1: Write `bench/hardware.sh`.** It's read-only. It prints:
  - `uname -a` and `/etc/os-release` `PRETTY_NAME`;
  - from `lscpu`: model name, cores and threads, and AVX2/AVX-512/AMX flags;
  - `free -g`, and `df -h ~ /tmp`;
  - `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` (or "no NVIDIA GPU"), and `rocm-smi` if present;
  - from `lspci`, any VGA, 3D or NPU line;
  - `python3 --version`, `ffmpeg -version | head -1`, and whether `systemctl --user` works.

- [ ] **Step 2: Run it remotely.** Run `ssh <user@host> 'bash -s' < bench/hardware.sh > bench/results/hardware.txt`. Expected: the file lists CPU, RAM, disk and GPU. If the connection is refused, report it to the user exactly as shown, and don't try other users or ports.

- [ ] **Step 3: Classify the tier and gate G2.**

  | Tier | Rule | Consequence |
  |---|---|---|
  | GPU | NVIDIA with ≥ 8 GB VRAM | Larger models are fine; speech-to-text on CUDA |
  | Strong CPU | ≥ 32 GB RAM, ≥ 8 cores | Whisper large-v3 int8 on CPU; Jais-ASR maybe |
  | Light CPU | Anything less | Whisper large-v3-turbo or smaller; first audio likely above 3 s |

  Tell the user the tier and what it means for the 2–3 s target, and wait for an OK. They may change hardware at this point.

- [ ] **Step 4: Commit**

```bash
git add bench/hardware.sh && git commit -m "bench: mini PC hardware report"
```

---

### Task 6: Speech-to-text benchmark on the mini PC (measurement)

**Files:**
- Create: `bench/stt_bench.py`, `bench/requirements-bench.txt` (`arabic-asr-benchmark[hf] @ git+https://github.com/echonova-ai-solutions/arabic-asr-benchmark`, `datasets[audio]`, `soundfile`), `bench/darija_prompts.txt` (only if Step 1 needs it)
- Output: `bench/results/stt/<model>.jsonl`, `bench/results/stt/refs_<set>.jsonl`, `bench/results/stt.md`

**Interfaces:**
- Prediction files use the benchmark format: `{"id": str, "prediction": str}`.
- References: `{"id": str, "dialect": str, "recording_id": str, "text": str}`, as in the benchmark repo's `examples/synthetic/references.jsonl`.

- [ ] **Step 1: Prepare the evaluation sets on the mini PC (30 clips each).**
  - **Egyptian and Saudi:** the first 30 clips of each from `echonova-ai/arabic-dialect-asr-benchmark` (split `test`, field `dialect` = `egyptian` / `saudi`). Read the dataset's license on its Hugging Face page first and note it. If it's gated, ask the user to accept it.
  - **Moroccan:** search Hugging Face for a Darija speech set that has human transcripts and a license allowing evaluation, and note its name and license. If none is usable, write 30 everyday Darija sentences to `bench/darija_prompts.txt` and have the user check them. Then ask the user to record themselves reading them. The sentences become the references.
  - **English:** 30 clips from `openslr/librispeech_asr` `test.clean`.

  Write `refs_<set>.jsonl` with exactly the clip ids used.

- [ ] **Step 2: Write `bench/stt_bench.py <model-id>`.** For each set and clip, it records the transcript and the wall time, and the detected language for multilingual models. Run each model warm: one throwaway clip first. Candidates, run in this order and only if Task 5's tier allows:
  1. `faster-whisper large-v3-turbo` (int8 on CPU, float16 on CUDA), `beam_size=1`, `vad_filter=True`, `language=None`.
  2. `faster-whisper large-v3`, same settings.
  3. `atlasia/moulsot.v0.3`, Moroccan set only. Follow its model card for the loading code; record its license.
  4. `IFM/Jais-ASR`, only if free RAM or VRAM is above 14 GB. It ships its own code (`trust_remote_code`): **read `modeling_jais_asr.py`, `processing_jais_asr.py` and `pipeline_jais_asr.py` first**, and record in the ledger that they were read.

- [ ] **Step 3: Score.** Arabic sets: `asr-bench evaluate --references refs_<set>.jsonl --predictions <model>.jsonl --system <model>`, which uses the default `echonova-v1` normalizer. English: compute WER with `editdistance` over lowercased, punctuation-stripped words.
- [ ] **Step 4: Write `bench/results/stt.md`.** One table: rows are models; columns are WER per set, median and p90 seconds per clip, real-time factor, peak RAM/VRAM, license, and the share of Arabic clips whose detected language was not `ar`. Add a 3-line takeaway.
- [ ] **Step 5: Commit the scripts.**

```bash
git add bench/stt_bench.py bench/requirements-bench.txt && git commit -m "bench: speech-to-text benchmark"
```

---

### Task 7: Voice samples on the mini PC (measurement → gate G3)

**Files:**
- Create: `bench/tts_samples.py`, `bench/tts_sentences.json`
- Output: `bench/results/tts/<dialect>/<engine>__<n>.wav`, `bench/results/tts.md`, `bench/results/tts/listen.html` (a local page with one `<audio>` per file, grouped by dialect)

- [ ] **Step 1: Write `bench/tts_sentences.json`.**

```json
{
  "en":    ["Hello, I'm Rawi. How can I help you today?", "It's a lovely day. What would you like to talk about?"],
  "ar-SA": ["هلا والله، أنا راوي. وش أقدر أساعدك فيه اليوم؟", "الجو اليوم حلو، تبي نسولف عن شي معين؟"],
  "ar-EG": ["أهلاً، أنا راوي. أقدر أساعدك في إيه النهارده؟", "الجو حلو النهارده، عايز نتكلم عن إيه؟"],
  "ar-MA": ["مرحبا، أنا راوي. كيفاش نقدر نعاونك اليوم؟", "الجو زوين اليوم، على أش بغيتي نهضرو؟"]
}
```

Also add the two DeepSeek replies per Arabic dialect that the user rated most natural in Task 4. Copy them from `bench/results/llm_dialects.md`.

- [ ] **Step 2: Install each candidate on the mini PC in a separate venv per engine** (`~/rawi-bench/<engine>/.venv`), to keep their dependencies from clashing. Record each license from its model card:
  - Piper `en_US-lessac-medium` (en), and Piper's Arabic voice if its voice list has one. That should be `ar_JO-kareem-medium`; confirm it in Piper's voice list. It's the MSA fallback for every Arabic dialect.
  - `oddadmix/lahgtna-chatterbox-v0` or its newer version: ar-SA, ar-EG, ar-MA. Use the newest version the model card points to. If it supports voice copying, also generate with one reference clip so all dialects share a voice. **Ask the user for the reference voice and their consent first, and don't use a real person's voice without it.**
  - `KandirResearch/DarijaTTS-v0.1-500M`: ar-MA.

- [ ] **Step 3: Write `bench/tts_samples.py <engine> <dialect>`.** It writes one WAV per sentence and records the time to synthesize each one, warm, as a JSON line.
- [ ] **Step 4: Write `bench/results/tts.md`** (engine × dialect: median seconds per sentence, real-time factor, license, sample rate) **and `listen.html`.**
- [ ] **Step 5: Gate G3.** Copy `bench/results/tts/` to the laptop (`scp -r`) and ask the user to open `listen.html` and rate each engine per dialect: natural / acceptable / wrong dialect / unusable. Record the ratings in `tts.md`.
- [ ] **Step 6: Commit the scripts.**

```bash
git add bench/tts_samples.py bench/tts_sentences.json && git commit -m "bench: voice samples per dialect"
```

---

### Task 8: Model decision record (gate G4)

**Files:**
- Create: `docs/superpowers/specs/2026-11-16-model-choices.md`
- Create: `voices.example.json` at the repo root. It's committed; the real `voices/voices.json` is created on the mini PC because `voices/` is git-ignored.

**Interfaces:**
- Produces, for Tasks 9, 10 and 13:
  - `STT_PRIMARY`, the model id and settings;
  - `STT_DARIJA`, a model id or `none`;
  - `DARIJA_ROUTING`, one of `none` / `classifier:<model>` / `both`;
  - the `voices.json` mapping, every dialect → `{"engine": str, ...engine settings}`;
  - the device, `cpu` or `cuda`;
  - the license per model.

- [ ] **Step 1: Write the decision record from the Task 4, 6 and 7 results.** For every stage, give the pick, the runner-up, the numbers that decided it, the license, and the user's rating. Apply approach C: keep the general model unless a dialect's result is "wrong" or "unusable" in the user's rating, or its WER is more than 10 points worse than the best specialist.
- [ ] **Step 2: Write `voices.example.json`** in the exact shape `tts.py` reads (Task 10):

```json
{
  "en":    {"engine": "piper", "model": "en_US-lessac-medium"},
  "ar-SA": {"engine": "<picked>", "model": "<id>"},
  "ar-EG": {"engine": "<picked>", "model": "<id>"},
  "ar-MA": {"engine": "<picked>", "model": "<id>"}
}
```

Engine-specific settings, such as a reference clip path or a speaker id, sit beside `model` in each object.

- [ ] **Step 3: Gate G4.** Show the user a one-screen summary: the model per stage, why, the license, the expected first-audio time per dialect from the measurements, and any dialect still marked weak. Wait for approval. If they change a pick, update the record.
- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-11-16-model-choices.md voices.example.json
git commit -m "docs: model choices for speech-to-text and voices"
```

---

### Task 9: Multilingual speech-to-text (`stt.py`)

**Files:**
- Modify: `stt.py`, `config.py`, `app.py` (`/api/transcribe` response), `.env.example`, `requirements.txt` (only if Task 8 picked a model faster-whisper can't run)
- Test: `tests/test_stt.py`

**Interfaces:**
- Consumes: Task 8's `STT_PRIMARY`, `STT_DARIJA` and `DARIJA_ROUTING`.
- Produces:
  - `@dataclass(frozen=True) class Transcript: text: str; language: Literal["en", "ar"]`.
  - `class Transcriber(Protocol): def transcribe(self, audio: bytes) -> Transcript`.
  - `get_transcriber() -> Transcriber`: cached; built from `config.STT_MODEL`, `config.STT_DEVICE` (`cpu` | `cuda` | `auto`) and `config.STT_COMPUTE` (default `int8` on CPU, `float16` on CUDA).
  - `transcribe_audio(data: bytes) -> Transcript`.
  - `POST /api/transcribe` returns `{"text": str, "language": "en" | "ar"}`, with the same 413/400/503 behaviour and copy as today.
  - Config: `STT_MODEL` (env, default from Task 8) replaces `WHISPER_MODEL`. The old `WHISPER_MODEL` is still read as a fallback, so an existing `.env` keeps working.

- [ ] **Step 1: Write the failing tests**

```python
def info(lang, probs): return NS(language=lang, all_language_probs=probs)

def fake_whisper(monkeypatch, runs):
    """runs: list of (segments, info) returned by successive transcribe() calls; records kwargs."""
    calls = []
    class M:
        def transcribe(self, audio, **kw):
            calls.append(kw); segs, inf = runs[len(calls) - 1]
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
    import sys, types
    seen = {}
    class WM:
        def __init__(self, name, **kw): seen.update(kw, name=name)
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WM))
    monkeypatch.setattr(config, "STT_DEVICE", "cuda"); monkeypatch.setattr(config, "STT_COMPUTE", "")
    stt.get_whisper.cache_clear()
    try: stt.get_whisper()
    finally: stt.get_whisper.cache_clear()
    assert seen["device"] == "cuda" and seen["compute_type"] == "float16" and seen["name"] == config.STT_MODEL
```

Also keep the existing 413, 400 and 503 tests, pointed at the new function names.

- [ ] **Step 2: Run the tests and watch them fail.** Run `.venv/bin/pytest tests/test_stt.py -q`. Expected: FAIL on `Transcript` being undefined and on `language="en"` still being passed.

- [ ] **Step 3: Implement.**
  - **Language detection:** the faster-whisper backend runs `transcribe(io.BytesIO(data), language=None, vad_filter=True, beam_size=1)` and consumes the segments.
    - If `info.language` is `en` or `ar`, return the result.
    - Otherwise pick whichever of `en` and `ar` has the higher probability in `info.all_language_probs`, with ties going to `ar`, and transcribe again with that language.
  - **Moroccan routing**, only if Task 8 picked a Darija model:
    - `classifier:<model>`: run the dialect classifier on the audio, and send Moroccan audio to the Darija backend instead.
    - `both`: run both transcribers in a thread pool and use the Darija transcript when the classifier or the text says Moroccan.

    Each routing mode gets its own failing test first, in the same style with fakes. If `DARIJA_ROUTING` is `none`, skip this part and record the ruling in the ledger.
  - `STT_DEVICE=auto` means `cuda` when `ctranslate2.get_cuda_device_count() > 0`, else `cpu`.

- [ ] **Step 4: Run both suites.** Expected: all pass.
- [ ] **Step 5: Check on this laptop with a real clip.** Record one English and one Arabic sentence through the page. The `/api/transcribe` responses should show `language` `en` and `ar` (DevTools Network tab).
- [ ] **Step 6: Commit**

```bash
git add stt.py config.py app.py .env.example requirements.txt tests/test_stt.py
git commit -m "feat: multilingual speech-to-text with en/ar language detection"
```

---

### Task 10: Voice per dialect (`tts.py`), `/api/speak` by dialect, health

**Files:**
- Modify: `tts.py`, `app.py`, `config.py`, `requirements.txt` (the engines Task 8 picked, pinned to exact versions)
- Test: `tests/test_tts.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: the `voices.json` shape from Task 8; `dialects.DIALECTS`.
- Produces:
  - `class VoiceEngine(Protocol): def synthesize(self, text: str) -> bytes  # WAV bytes`
  - `ENGINES: dict[str, Callable[[dict], VoiceEngine]]`: always `"piper"`, plus one adapter per engine Task 8 picked (for example `"chatterbox"`, `"outetts"`).
  - `config.VOICES_FILE` (env, default `voices/voices.json`).
  - `load_voice_map(path: Path) -> dict[str, dict]`: reads JSON, keeps only keys in `DIALECTS`, and raises `ValueError` on a dialect whose engine isn't in `ENGINES`. Without the file, it returns `{"en": {"engine": "piper", "model": config.PIPER_VOICE}}`, so today's setup keeps working.
  - `get_engine(dialect: str) -> VoiceEngine`: cached per dialect; raises `KeyError` when the dialect has no voice.
  - `synthesize(text: str, dialect: str) -> bytes`.
  - `available_dialects() -> list[str]`: the dialects in the voice map whose model files are present.
  - `POST /api/speak` with body `{"text": str (1..1000), "dialect": Dialect}` returns `audio/wav`. Errors:
    - 422 for an invalid dialect (Literal validation);
    - 503 `"No voice for this dialect."`;
    - 503 `"Voice unavailable."` when loading fails;
    - 500 `"Speech synthesis failed."`.
  - `GET /api/voices` is removed.
  - `GET /api/health` returns `{"deepseek_key": bool, "voices": {"en": bool, "ar-MA": bool, "ar-EG": bool, "ar-SA": bool}}`.
  - `preload_models()` loads the transcriber, then every configured voice. It prints `Loading <what>…` for each and logs failures without stopping.

- [ ] **Step 1: Write the failing tests**

```python
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
    def none(d): raise KeyError(d)
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
```

Keep and adapt the existing tests:
- load failure → 503 `"Voice unavailable."`;
- synthesis failure → 500;
- preload survives failures;
- the cross-site 403 tests, now posting a `dialect`.

For each non-Piper adapter, add one test with the engine's library faked through `sys.modules`, the same pattern as `test_model_settings_cuda`. It asserts the adapter passes the configured model id and device, and returns WAV bytes starting with `b"RIFF"`.

- [ ] **Step 2: Run the tests and watch them fail.** Run `.venv/bin/pytest -q`. Expected: FAIL on the missing `load_voice_map`, `get_engine` and `dialect` field.
- [ ] **Step 3: Implement.** Each adapter imports its library lazily inside its constructor, as `get_voice` does today, so importing `tts` stays light. It converts whatever the engine returns (a numpy array, or a sample rate plus tensor) to 16-bit mono WAV with `wave`.
- [ ] **Step 4: Run both suites.** Expected: all pass.
- [ ] **Step 5: Commit**

```bash
git add tts.py app.py config.py requirements.txt tests
git commit -m "feat: one voice per dialect, /api/speak by dialect, voices in health"
```

---

### Task 11: Page: RTL, dialect badge, Arabic splitting, voice per dialect, offline message

**Files:**
- Create: `static/dialects.js`, `tests/dialects.test.mjs`
- Modify: `static/speech.js`, `static/app.js`, `static/index.html`, `static/app.css`, `tests/speech.test.mjs`

**Interfaces:**
- Consumes: the SSE contract from Task 3, `/api/transcribe` `{text, language}` from Task 9, and `/api/speak` `{text, dialect}` and `/api/health` from Task 10.
- Produces (`static/dialects.js`):
  - `DIALECTS = { en: { label: 'English', native: 'English', dir: 'ltr', lang: 'en' }, 'ar-MA': { label: 'Moroccan', native: 'الدارجة', dir: 'rtl', lang: 'ar-MA' }, 'ar-EG': { label: 'Egyptian', native: 'المصري', dir: 'rtl', lang: 'ar-EG' }, 'ar-SA': { label: 'Saudi', native: 'السعودي', dir: 'rtl', lang: 'ar-SA' } }`
  - `hasArabic(text) -> boolean` (same ranges as `dialects.has_arabic`).
  - `languageOf(text) -> 'en' | 'ar'`.
- Produces (`static/speech.js`): `SentenceBuffer` also splits on `؟`, and on `.`/`!`/`?` followed by Arabic text. `voiceLabel` is removed.

- [ ] **Step 1: Write the failing node tests**

`tests/speech.test.mjs` (add):

```js
test('splits on Arabic question mark', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('وش أخبارك؟ أنا '), ['وش أخبارك؟']);
  assert.deepEqual(b.flush(), ['أنا']);
});
test('Arabic comma does not split', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('هلا والله، كيف حالك؟ '), ['هلا والله، كيف حالك؟']);
});
test('decimal number does not split', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('It costs 3.5 riyals. Ok'), ['It costs 3.5 riyals.']);
});
test('period before Arabic splits', () => {
  const b = new SentenceBuffer();
  assert.deepEqual(b.push('تمام. شكراً '), ['تمام.']);
});
```

Note: the existing abbreviation rule splits only before an uppercase letter. Arabic letters have no case, so Arabic text after `.` must still split. Only a Latin letter after an abbreviation waits.

`tests/dialects.test.mjs`:

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { DIALECTS, hasArabic, languageOf } from '../static/dialects.js';
test('four dialects with direction', () => {
  assert.deepEqual(Object.keys(DIALECTS), ['en', 'ar-MA', 'ar-EG', 'ar-SA']);
  assert.equal(DIALECTS['ar-MA'].dir, 'rtl'); assert.equal(DIALECTS.en.dir, 'ltr');
});
test('language of typed text', () => {
  assert.equal(languageOf('عندي meeting'), 'ar'); assert.equal(languageOf('hello'), 'en');
  assert.ok(hasArabic('هلا') && !hasArabic('3.5'));
});
```

Delete the `voiceLabel` test.

- [ ] **Step 2: Run the tests and watch them fail.** Run `node --test tests/*.test.mjs`. Expected: FAIL. `؟` doesn't split, and `dialects.js` is missing. The decimal test may already pass because the `\s+` rule needs whitespace after the `.`; if so, it stays as a regression guard and the ledger notes it.
- [ ] **Step 3: Implement `dialects.js` and the splitter changes.** The sentence-end regex becomes `/[.!?؟](\s+)(\S)?/g`. The abbreviation wait only holds when the next char is a Latin letter.
- [ ] **Step 4: Update `app.js`.**
  - `ask(question, sentAt, language)` sends `{question, history, language}`, where `language` comes from the transcript or `languageOf(question)` for typed text.
  - On `dialect`, store the turn's dialect. Set `bubble.dir` and `bubble.lang` from `DIALECTS`, and add a badge `<span class="dialect-badge">` showing `native · label` (for example "الدارجة · Moroccan") before the bubble text.
  - User bubbles get `dir="auto"`.
  - `speak(sentence)` posts `{text, dialect}` with the current turn's dialect. A sentence that arrives before a `dialect` event waits in the queue until it arrives. The server always sends `dialect` first, so this is only defensive.
  - On `error`:
    - show `data.message`;
    - if `kind === 'offline'`, also speak it with dialect `data.language === 'ar' ? 'ar-SA' : 'en'`, which uses local voices and works offline;
    - if `kind === 'nokey'`, show it only.
  - **Remove** the Settings button, dialog, `initSettings`, `voiceGroup`, the voice localStorage, the Ctrl/Cmd+, shortcut, and the `NO_VOICE` startup message. The voice is now picked by dialect.
  - `checkHealth` shows "No DeepSeek key" if the key is missing, else "Voice missing: Moroccan" (listing the labels of dialects whose voice is `false`), else "Ready".
  - The history and the 6-message cap stay as today.
- [ ] **Step 5: Update `index.html` and `app.css`.**
  - Title and `<h1>` become "Rawi". Empty hint: "Talk to me in English, الدارجة, المصري or السعودي."
  - Add the Arabic font stack `--font-ar: "Noto Naskh Arabic", "Geeza Pro", "Segoe UI", system-ui, sans-serif`, applied to `[dir=rtl]`. It's local system fonts only, with no font CDN, so the page works offline.
  - The badge uses the existing small caption style and sits on the reading start edge. RTL bubbles align with logical properties (`margin-inline-start`).
  - Remove the settings-dialog CSS.
  - Keep text contrast ≥ 4.5:1, as verified in the HIG work.
- [ ] **Step 6: Run both suites.** Expected: all pass.
- [ ] **Step 7: Browser check** (chrome-devtools, this laptop, real DeepSeek, local voices). Type the Task 4 `switch` questions in order. Expected:
  - each reply shows the right badge and direction;
  - English text is left-aligned and Arabic right-aligned;
  - speech requests carry the matching `dialect` (Network tab);
  - no console errors.

  Then restart the server once with `DEEPSEEK_BASE_URL=http://127.0.0.1:9` and ask in Arabic. Expected: the Arabic offline message is shown and spoken in the Arabic voice.
- [ ] **Step 8: Commit**

```bash
git add static tests
git commit -m "feat: page speaks each reply in its dialect's voice; RTL, dialect badge, offline message"
```

---

### Task 12: LAN access, HTTPS option, stage timing logs

**Files:**
- Modify: `config.py`, `app.py`, `.env.example`
- Test: `tests/test_app.py`

**Interfaces:**
- Produces, as config:
  - `HOST` (default `127.0.0.1`);
  - `ALLOWED_HOSTS: list[str]` (comma list, default `127.0.0.1,localhost`);
  - `SSL_CERTFILE` and `SSL_KEYFILE` (default empty);
  - `allowed_origins() -> set[str]`: for each allowed host, `http://h:PORT`, plus `https://h:PORT` when TLS is set;
  - `uvicorn_ssl_kwargs() -> dict[str, str]`: both ssl args when both are set, else `{}`.
- `SameOriginMiddleware` uses `config.allowed_origins()`, and `TrustedHostMiddleware` uses `config.ALLOWED_HOSTS`.
- `uvicorn.run(app, host=config.HOST, port=config.PORT, **config.uvicorn_ssl_kwargs())`.
- Timing log: one line per answer, `timing stt_ms=… llm_first_ms=…`, at `INFO`, with no content. The server logs `stt_ms` (in `/api/transcribe`) and `llm_first_ms` (time from the request to the first `delta`). The page keeps logging `first audio after` in the console; Task 14 reads both.

- [ ] **Step 1: Write the failing tests**

```python
import importlib

@pytest.fixture
def reload_config(monkeypatch):
    yield lambda: importlib.reload(config)
    monkeypatch.undo(); importlib.reload(config); importlib.reload(app_module)

def test_lan_host_allowed(monkeypatch, reload_config):
    monkeypatch.setenv("ALLOWED_HOSTS", "127.0.0.1,localhost,rawi.local")
    reload_config(); app = importlib.reload(app_module).app
    assert TestClient(app, base_url="http://rawi.local").get("/").status_code == 200
    assert TestClient(app, base_url="http://evil.example").get("/").status_code == 400

def test_origins_follow_hosts_and_tls(monkeypatch, reload_config):
    monkeypatch.setenv("ALLOWED_HOSTS", "rawi.local")
    monkeypatch.setenv("SSL_CERTFILE", "c.pem"); monkeypatch.setenv("SSL_KEYFILE", "k.pem")
    cfg = reload_config()
    assert cfg.allowed_origins() == {f"http://rawi.local:{cfg.PORT}", f"https://rawi.local:{cfg.PORT}"}

def test_ssl_settings_from_env(monkeypatch, reload_config):
    monkeypatch.setenv("SSL_CERTFILE", "c.pem"); monkeypatch.setenv("SSL_KEYFILE", "k.pem")
    assert reload_config().uvicorn_ssl_kwargs() == {"ssl_certfile": "c.pem", "ssl_keyfile": "k.pem"}
    monkeypatch.delenv("SSL_KEYFILE")
    assert reload_config().uvicorn_ssl_kwargs() == {}

def test_default_bind_is_loopback(monkeypatch, reload_config):
    monkeypatch.delenv("HOST", raising=False)
    assert reload_config().HOST == "127.0.0.1"
```

Note: `load_dotenv` doesn't override variables already set, so these env-based tests are unaffected by a local `.env` only for keys the test sets. If a developer's `.env` sets `HOST`, `test_default_bind_is_loopback` deletes it from the environment first; `load_dotenv` would then re-add it from `.env`. Guard against that by making `config` read `.env` with `load_dotenv(..., override=False)` (the default) and having the test set `HOST` explicitly to an empty string, which config treats as unset (`os.getenv("HOST") or "127.0.0.1"`).

- [ ] **Step 2: Run the tests and watch them fail.** Expected: FAIL. There's no `ALLOWED_HOSTS`, `allowed_origins` or `uvicorn_ssl_kwargs`.
- [ ] **Step 3: Implement.** Add the config readers and the middleware wiring, and add the timing log with `time.perf_counter()`.
- [ ] **Step 4: Run both suites.** Expected: all pass.
- [ ] **Step 5: Commit**

```bash
git add config.py app.py .env.example tests
git commit -m "feat: opt-in LAN serving with HTTPS, per-stage timing log"
```

---

### Task 13: Deploy to the mini PC

**Files:**
- Create: `deploy/install.sh`, `deploy/rawi.service`, `deploy/README.md`

**Interfaces:**
- Consumes: Task 8's model choices and `voices.example.json`; Task 12's config.

- [ ] **Step 1: Write `deploy/install.sh`.** It's idempotent and runs on the mini PC as the user, with no sudo. It:
  - creates `.venv` in `~/rawi` if it's absent;
  - installs `requirements.txt` with `TMPDIR=~/.cache/pip-tmp`, plus the CUDA runtime wheels for ctranslate2 only if `nvidia-smi` exists;
  - downloads the chosen speech-to-text model (a warm-up load) and the voices into `voices/`;
  - copies `voices.example.json` to `voices/voices.json` if it's absent;
  - prints the next manual step: "create ~/rawi/.env with DEEPSEEK_API_KEY=…".

  **The script never copies or creates the key.**
- [ ] **Step 2: Write `deploy/rawi.service`**, a systemd **user** unit:
  - `WorkingDirectory=%h/rawi`
  - `ExecStart=%h/rawi/.venv/bin/python app.py`
  - `Restart=on-failure`

  The app reads `.env` itself. In `deploy/README.md`, document `systemctl --user enable --now rawi`, and `sudo loginctl enable-linger <user>` (the user runs this one, since it needs sudo) so the service survives logout and reboot.
- [ ] **Step 3: Copy and install.** First run `rsync -a --exclude .venv --exclude .env --exclude voices --exclude bench/results --exclude .git ./ <user@host>:rawi/`, then `ssh <user@host> 'bash ~/rawi/deploy/install.sh'`. Expected: it ends with the `.env` instruction. Then ask the user to create `.env` on the mini PC themselves, for example by typing `! ssh <user@host>` and using an editor there. Don't handle the key.
- [ ] **Step 4: Choose the access path.** Recommended: an SSH port-forward from the demo laptop, `ssh -N -L 8765:127.0.0.1:8765 <user@host>`. The server stays bound to `127.0.0.1`, and the browser opens `http://localhost:8765`, so the mic works.
  - **Tablet alternative:** `mkcert` creates a cert for the mini PC's hostname, set through `SSL_CERTFILE`, `SSL_KEYFILE`, `HOST=0.0.0.0` and `ALLOWED_HOSTS=<hostname>`. The user installs mkcert's root CA on the tablet.
  - **Ask the user before binding to `0.0.0.0`.** It exposes the app to the whole network.
- [ ] **Step 5: Smoke test.** On the mini PC, `curl -s http://127.0.0.1:8765/api/health`. Expected: `deepseek_key: true` and every dialect `true` in `voices`. Through the tunnel, open the page on the laptop and say "Hi". Expected: a spoken English reply.
- [ ] **Step 6: Commit**

```bash
git add deploy && git commit -m "deploy: mini PC install script and user service"
```

---

### Task 14: Rehearsal harness, measurements, README and runbook (gate G5)

**Files:**
- Create: `bench/rehearsal.py`, `docs/demo-runbook.md`
- Modify: `README.md`, `.gitignore` (add `bench/demo_clips/`)
- Output: `bench/results/rehearsal.md`

- [ ] **Step 1: Record the demo questions.** Ask the user to record their rehearsed demo questions, or to provide audio files. Aim for two per dialect, two mixed, and the `switch` sequence. Save them on the mini PC in `bench/demo_clips/<n>_<dialect>.<ext>`; the file name carries the expected dialect.
- [ ] **Step 2: Write `bench/rehearsal.py`.** For each clip, in order and with history:
  1. POST it to `/api/transcribe` and time it;
  2. POST the text to `/api/ask` and time the first `delta`;
  3. POST the first full sentence (split with the same rule as `speech.js`) to `/api/speak` with the received dialect, and time it.

  Record the transcript, the detected language, the dialect tag, the stage times, and **first audio ≈ stt + time to first sentence + tts_first**. Write `bench/results/rehearsal.md` with one row per clip, medians per dialect, and a pass mark for tag = expected dialect.
- [ ] **Step 3: Run it on the mini PC three times.** Expected: all tags match. Report the median first audio per dialect against the 2–3 s target. A dialect over 4 s goes to the user with the slowest stage named. Don't tune silently.
- [ ] **Step 4: Run the offline check.** With the mini PC's internet cut (the user unplugs it, or runs `nmcli networking off` themselves), ask one question. Expected: the spoken offline message within about 10 s. DeepSeek's connect timeout is 5 s with 1 retry. If that's too slow for the demo, propose `max_retries=0`, and decide with the user.
- [ ] **Step 5: Write `docs/demo-runbook.md`.**
  - **The day before:** sync and run `install.sh` again if the code changed, run the rehearsal script, and charge the laptop.
  - **At the venue:**
    1. internet check;
    2. `systemctl --user status rawi`;
    3. start the tunnel;
    4. open `http://localhost:8765`;
    5. one test sentence per dialect;
    6. headset or mic placement.
  - **If something breaks:**
    - no reply → the offline message is expected;
    - the wrong dialect → ask again with a longer sentence;
    - silence → check the voices in the status pill.
  - **The demo script:** the `switch` sequence.
- [ ] **Step 6: Update `README.md` for Rawi.**
  - What it does. What leaves the machine: only question and reply text, to DeepSeek.
  - Setup: laptop dev and mini PC via `deploy/`.
  - The config vars: `STT_MODEL`, `STT_DEVICE`, `STT_COMPUTE`, `VOICES_FILE`, `HOST`, `ALLOWED_HOSTS`, `SSL_CERTFILE`, `SSL_KEYFILE`, `DEEPSEEK_BASE_URL`.
  - The measured performance table from `rehearsal.md`.
  - Troubleshooting for every user-facing message: `NO_KEY`, `LLM_FAILED`, `OFFLINE`, "No voice for this dialect.", "Voice unavailable.", "Speech synthesis failed.", and the mic messages.
  - Remove all ciso-rag content.
- [ ] **Step 7: Gate G5.** The user runs the demo script live through the tunnel and confirms it's demo-ready. Record any issue as a new task rather than fixing it silently.
- [ ] **Step 8: Commit**

```bash
git add bench/rehearsal.py docs/demo-runbook.md README.md .gitignore && git commit -m "docs: rehearsal harness, demo runbook and README for Rawi"
```

---

## Spec coverage

| Spec item | Task |
|---|---|
| Hear the four languages | 9 |
| Reply in the same dialect | 3 |
| Voice per dialect | 10, 11 |
| Mixed questions | 3 (prompt), 4 (sample) |
| Normal talk only | 1 |
| Audio stays local | 9, 10, Global Constraints |
| 2–3 s | 12 (timing), 14 (measure) |
| Anti-goal: wrong dialect | 3, 4, 14 |
| Anti-goal: silence | 3, 11, 14 |
| Anti-goal: long answers | 3 |
| Mini PC | 5, 13 |
| Open licenses | 6, 7, 8 |
| Key handling | 4, 13 |
| LAN | 12, 13 |
| Tap to talk, interrupt | 11 (existing mic behaviour kept) |
| Topics, persona | 3 |
| Dialect detection | 3, 9 |
| UI | 11 |
| Code split | 2 |
| Phase 1 / 2 / 3 | 4–8 / 9–12 / 13–14 |
| Checks | 14 |

**Signatures used across tasks:**
- `llm.stream_turn(messages)`
- `llm.DialectTagParser.feed/finish`
- `llm.OFFLINE`, `llm.OfflineError`
- `stt.Transcript(text, language)`, `stt.transcribe_audio`, `stt.get_transcriber`, `stt.get_whisper`
- `tts.get_engine(dialect)`, `tts.synthesize(text, dialect)`, `tts.available_dialects()`, `tts.load_voice_map(path)`
- `config.allowed_origins()`, `config.uvicorn_ssl_kwargs()`
- `dialects.DIALECTS`, `DEFAULT_ARABIC`, `has_arabic`, `guess_dialect`, `language_of`

**Ruling recorded in the plan:** Task 11 removes the Settings sheet (gear, dialog, Ctrl/Cmd+,), because the voice now follows the dialect. Removing it was offered earlier in the project.
