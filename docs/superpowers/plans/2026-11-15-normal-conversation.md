# Normal Conversation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the voice assistant hold a normal conversation, so that DeepSeek searches the knowledge base only when a question is about the user's documents, and flags any answer that comes from general knowledge.

**Architecture:** `/api/ask` stops searching first. It calls DeepSeek with a `search_knowledge_base` tool and streams its reply. When DeepSeek calls the tool, the server runs `search_rag`, sends a `sources` event, returns the passages as the tool result, and streams the follow-up answer. Each answer allows at most 2 searches. The page appends sources when more than one search ran.

**Tech Stack:** Existing: FastAPI, openai SDK pointed at DeepSeek (OpenAI-style function calling), pytest, vanilla JS.

**Spec:** `docs/superpowers/specs/2026-11-15-normal-conversation-design.md`. It builds on `docs/superpowers/specs/2026-11-15-whisper-voice-assistant-design.md`.

## Global Constraints

- Work on a new branch `feat/normal-conversation` off `main`.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Small talk and general questions must never call `search_rag`, so no passages reach DeepSeek.
- The classification setting still applies to every search through `search_rag(query, body.classification)`. That path is fail-closed and must stay unchanged.
- The cap is at most **2** searches per answer. Once the cap is reached, the tool is not offered again.
- Error strings stay unchanged: `RAG_DOWN`, `NO_KEY`, `LLM_FAILED`. `NOT_FOUND` is removed.
- SSE events are unchanged except that `sources` becomes optional and may arrive 0, 1 or 2 times per answer.
- The DeepSeek key is never logged or sent to the page. Never read `.env`.
- Test commands: `.venv/bin/pytest -q` and `node --test tests/*.test.mjs`.

## Review Focus

1. **DeepSeek sends malformed or empty tool arguments.** The search falls back to the user's question instead of crashing. Test in Task 1 (`query=""`) and Task 2 (empty query uses the question).
2. **DeepSeek calls the tool repeatedly, or several times in one turn.** Real searches stop at 2. Every extra call still gets a tool reply (`SEARCH_LIMIT`), and the answer finishes. Tests in Task 2.
3. **The tunnel is down.** Small talk still works because it never touches ciso-rag. A document question gets `RAG_DOWN`. Tests in Task 2.
4. **Two searches in one answer.** The page shows both sets of sources instead of the second replacing the first. Manual check in Task 3.
5. **DeepSeek says text such as "Let me check." before calling the tool.** That text is forwarded and spoken, and the final answer still follows. Test in Task 2.

---

## File Structure

- `app.py`: prompt section (new `SYSTEM_PROMPT`, `SEARCH_TOOL`, `format_passages`, changed `build_messages`), DeepSeek section (`ToolCall` and `stream_turn` replace `stream_answer`), and `ask_events`.
- `tests/test_app.py`: replace the ask, build_messages and stream_answer tests.
- `static/app.js`: `renderSources` appends instead of replacing.
- `README.md`: behaviour and performance notes.

---

### Task 1: Tool-calling primitives

**Files:**
- Modify: `app.py` (prompt and DeepSeek sections)
- Test: `tests/test_app.py`

**Interfaces:**
- Produces:
  - `SYSTEM_PROMPT: str` (new text, below).
  - `SEARCH_TOOL: dict`, an OpenAI function-tool schema with name `search_knowledge_base`, one required string parameter `query`, and the description from the spec: "Search the user's CISO knowledge base: internal policies, NCA ECC, SAMA CSF, PDPL, ISO 27001, NIST CSF, estate architecture, project plans, Apple HIG. Use a complete, self-contained query."
  - `def format_passages(passages: list[dict]) -> str` returns the numbered `[n] {file_name} — {header_path}\n{text}` blocks joined by blank lines. It uses `.get(..., "")` for every field and returns `NO_PASSAGES` when the list is empty.
  - `NO_PASSAGES = "No matching passages found."`
  - `SEARCH_LIMIT = "Search limit reached. Answer with what you have."`
  - `def build_messages(question: str, history: list[dict]) -> list[dict]` returns `[system] + history[-6:] + [{"role": "user", "content": question}]`. The passages argument is removed.
  - `@dataclass(frozen=True) class ToolCall: id: str; query: str`
  - `async def stream_turn(messages: list[dict], tools: list[dict] | None) -> AsyncIterator[str | ToolCall]`. It calls `get_deepseek().chat.completions.create(model=DEEPSEEK_MODEL, messages=messages, stream=True, **({"tools": tools} if tools else {}))`. It yields each non-empty `delta.content` string as it arrives. It accumulates `delta.tool_calls` fragments by `index` (`id` and `function.name` arrive in the first fragment, and `function.arguments` concatenates across fragments). After the stream ends, it yields one `ToolCall` per accumulated call, in index order. The `query` comes from `json.loads(arguments)["query"]`, or `""` if the JSON is invalid or the key is missing. The stream is closed in `finally`.
  - Remove `stream_answer` and `NOT_FOUND`.

`SYSTEM_PROMPT`:
```
You are a friendly voice assistant for a CISO. You can chat normally and answer general questions. For anything about the user's own documents, internal policies, regulations (NCA ECC, SAMA CSF, PDPL, ISO 27001, NIST CSF), architecture, projects or the Apple HIG, call search_knowledge_base first. When you answer from search results, use only what the passages say. When you answer from general knowledge, start by saying it is not from their documents. Reply in 2 to 4 short sentences of plain spoken English. No markdown, no lists, no headings, and do not read out document names or passage numbers.
```

- [ ] **Step 1: Branch**

Run: `git switch -c feat/normal-conversation`

- [ ] **Step 2: Replace the failing tests**

Delete `test_build_messages_shape`, `test_build_messages_tolerates_missing_fields` and `test_stream_answer_wiring`. Add the tests below.

```python
from types import SimpleNamespace as NS

def test_format_passages():
    p = {"file_name": "P-03.md", "header_path": "Access > MFA", "text": "MFA is required."}
    assert app_module.format_passages([p, {"classification": "PUBLIC"}]) == \
        "[1] P-03.md — Access > MFA\nMFA is required.\n\n[2]  — \n"
    assert app_module.format_passages([]) == app_module.NO_PASSAGES

def test_build_messages_shape():
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(8)]
    msgs = app_module.build_messages("Hi there", hist)
    assert msgs[0] == {"role": "system", "content": app_module.SYSTEM_PROMPT}
    assert [m["content"] for m in msgs[1:-1]] == ["2", "3", "4", "5", "6", "7"]
    assert msgs[-1] == {"role": "user", "content": "Hi there"}

def fake_deepseek(monkeypatch, chunks):
    seen = {"closed": 0}
    class Stream:
        def __aiter__(self):
            async def gen():
                for c in chunks: yield c
            return gen()
        async def close(self): seen["closed"] += 1
    async def create(**kw):
        seen["kw"] = kw; return Stream()
    monkeypatch.setattr(app_module, "get_deepseek",
                        lambda: NS(chat=NS(completions=NS(create=create))))
    return seen

def text(c): return NS(choices=[NS(delta=NS(content=c, tool_calls=None))])
def tc(index, id=None, name=None, args=""):
    return NS(choices=[NS(delta=NS(content=None, tool_calls=[
        NS(index=index, id=id, function=NS(name=name, arguments=args))]))])

def collect(messages, tools):
    async def run(): return [x async for x in app_module.stream_turn(messages, tools)]
    import asyncio; return asyncio.run(run())

def test_stream_turn_text_then_tool_call(monkeypatch):
    seen = fake_deepseek(monkeypatch, [
        text("Let me check. "),
        tc(0, id="call_1", name="search_knowledge_base", args='{"que'),
        tc(0, args='ry": "MFA policy"}'),
        NS(choices=[]),
    ])
    out = collect([{"role": "user", "content": "q"}], [app_module.SEARCH_TOOL])
    assert out == ["Let me check. ", app_module.ToolCall(id="call_1", query="MFA policy")]
    assert seen["kw"]["tools"] == [app_module.SEARCH_TOOL]
    assert seen["kw"]["stream"] is True and seen["kw"]["model"] == app_module.DEEPSEEK_MODEL
    assert seen["closed"] == 1

def test_stream_turn_two_calls_in_index_order(monkeypatch):
    fake_deepseek(monkeypatch, [
        tc(1, id="b", name="search_knowledge_base", args='{"query": "SAMA"}'),
        tc(0, id="a", name="search_knowledge_base", args='{"query": "NCA"}'),
    ])
    assert collect([], [app_module.SEARCH_TOOL]) == [
        app_module.ToolCall("a", "NCA"), app_module.ToolCall("b", "SAMA")]

def test_stream_turn_bad_arguments_give_empty_query(monkeypatch):
    fake_deepseek(monkeypatch, [tc(0, id="x", name="search_knowledge_base", args="not json")])
    assert collect([], [app_module.SEARCH_TOOL]) == [app_module.ToolCall("x", "")]

def test_stream_turn_without_tools_omits_param(monkeypatch):
    seen = fake_deepseek(monkeypatch, [text("Hello.")])
    assert collect([], None) == ["Hello."]
    assert "tools" not in seen["kw"]
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_app.py -k "format_passages or build_messages or stream_turn" -v`
Expected: FAIL. `format_passages`, `stream_turn` and `ToolCall` are not defined, and `build_messages` has the wrong arity.

- [ ] **Step 4: Implement the Task 1 interfaces in `app.py`**

Leave `ask_events` alone except where it must still import and run. The ask tests are rewritten in Task 2, so the existing `test_ask_*` tests can fail until then. Mark them with `@pytest.mark.skip(reason="rewritten in Task 2")` for this commit.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/bin/pytest -q`
Expected: all pass, with the old ask tests skipped.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py && git commit -m "feat: DeepSeek tool-calling primitives for knowledge-base search"
```

---

### Task 2: `/api/ask` conversation loop

**Files:**
- Modify: `app.py` (`ask_events`)
- Test: `tests/test_app.py` (delete the skipped old ask tests and add the ones below; keep `test_ask_no_key` and `test_ask_validation`)

**Interfaces:**
- Consumes: `stream_turn`, `ToolCall`, `SEARCH_TOOL`, `build_messages`, `format_passages`, `NO_PASSAGES`, `SEARCH_LIMIT` (Task 1), and `search_rag` (existing).
- Produces:
  - `MAX_SEARCHES = 2`
  - `ask_events(body)` with this behaviour:
    1. If no key, send `error` NO_KEY.
    2. Set `messages = build_messages(question, history)` and `searches = 0`.
    3. Loop for up to `MAX_SEARCHES + 1` turns. Each turn calls `stream_turn(messages, [SEARCH_TOOL] if searches < MAX_SEARCHES else None)`. Forward each `str` as `delta` and collect each `ToolCall`.
    4. If a turn has no calls, send `done` and return.
    5. Otherwise, append `{"role": "assistant", "content": <this turn's text or "">, "tool_calls": [{"id", "type": "function", "function": {"name": "search_knowledge_base", "arguments": json.dumps({"query": q})}} ...]}`.
    6. For each call, in order: if `searches >= MAX_SEARCHES`, the tool content is `SEARCH_LIMIT`. Otherwise increment `searches`, run `search_rag(call.query or body.question, body.classification)` (on `httpx.HTTPError` or `ValueError`, send `error` RAG_DOWN and return), send a `sources` event (same shape as today), and set the tool content to `format_passages(passages)`. Append `{"role": "tool", "tool_call_id": call.id, "content": content}`.
    7. Any exception from `stream_turn` logs the type name and sends `error` LLM_FAILED. The existing `except` handling keeps the same shape.
    8. If the loop runs out of turns, send `done`. This can only happen if DeepSeek emits a tool call even though no tools were offered, and any such call is ignored.

- [ ] **Step 1: Write the failing tests**

```python
def fake_turns(monkeypatch, *turns):
    calls = []
    async def turn(messages, tools):
        i = len(calls)
        calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        for item in turns[i]:
            if isinstance(item, Exception): raise item
            yield item
    monkeypatch.setattr(app_module, "stream_turn", turn)
    return calls

def fake_search(monkeypatch, result=None, exc=None):
    seen = []
    async def s(q, c):
        seen.append((q, c))
        if exc: raise exc
        return [P("INTERNAL")] if result is None else result
    monkeypatch.setattr(app_module, "search_rag", s)
    return seen

TC = app_module.ToolCall

def test_ask_small_talk_never_searches(monkeypatch, keyed):
    calls = fake_turns(monkeypatch, ["Hi! ", "How can I help?"])
    seen = fake_search(monkeypatch)
    assert events(ask(question="hello")) == [("delta", {"text": "Hi! "}), ("delta", {"text": "How can I help?"}), ("done", {})]
    assert seen == [] and calls[0]["tools"] == [app_module.SEARCH_TOOL]

def test_ask_small_talk_works_with_tunnel_down(monkeypatch, keyed):
    fake_turns(monkeypatch, ["Hello."])
    fake_search(monkeypatch, exc=httpx.ConnectError("refused"))
    assert events(ask(question="hi")) == [("delta", {"text": "Hello."}), ("done", {})]

def test_ask_document_question_searches(monkeypatch, keyed):
    calls = fake_turns(monkeypatch, ["Let me check. ", TC("c1", "MFA policy")], ["MFA is required."])
    seen = fake_search(monkeypatch)
    ev = events(ask(classification="exclude_sensitive"))
    assert [e for e, _ in ev] == ["delta", "sources", "delta", "done"]
    assert ev[1][1] == [{"file_name": "doc.md", "header_path": "A > B", "classification": "INTERNAL", "text": "t"}]
    assert seen == [("MFA policy", "exclude_sensitive")]
    assistant, tool = calls[1]["messages"][-2:]
    assert assistant["role"] == "assistant" and assistant["content"] == "Let me check. "
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"query": "MFA policy"}
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": "[1] doc.md — A > B\nt"}

def test_ask_empty_query_uses_question(monkeypatch, keyed):
    fake_turns(monkeypatch, [TC("c1", "")], ["ok"])
    seen = fake_search(monkeypatch)
    events(ask())
    assert seen == [("Is MFA required?", "all")]

def test_ask_empty_search_tells_model(monkeypatch, keyed):
    calls = fake_turns(monkeypatch, [TC("c1", "x")], ["Not in your documents, but generally..."])
    fake_search(monkeypatch, result=[])
    assert [e for e, _ in events(ask())] == ["sources", "delta", "done"]
    assert calls[1]["messages"][-1]["content"] == app_module.NO_PASSAGES

def test_ask_search_cap_across_turns(monkeypatch, keyed):
    calls = fake_turns(monkeypatch, [TC("a", "1")], [TC("b", "2")], ["done talking"])
    seen = fake_search(monkeypatch)
    assert [e for e, _ in events(ask())] == ["sources", "sources", "delta", "done"]
    assert len(seen) == 2 and calls[2]["tools"] is None

def test_ask_search_cap_within_one_turn(monkeypatch, keyed):
    calls = fake_turns(monkeypatch, [TC("a", "1"), TC("b", "2"), TC("c", "3")], ["answer"])
    seen = fake_search(monkeypatch)
    events(ask())
    assert len(seen) == 2
    tools_msgs = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools_msgs] == ["a", "b", "c"]
    assert tools_msgs[2]["content"] == app_module.SEARCH_LIMIT
    assert calls[1]["tools"] is None

def test_ask_rag_down_during_search(monkeypatch, keyed):
    fake_turns(monkeypatch, [TC("c1", "q")])
    fake_search(monkeypatch, exc=httpx.ConnectError("refused"))
    assert events(ask()) == [("error", {"message": app_module.RAG_DOWN})]

def test_ask_rag_not_json_during_search(monkeypatch, keyed):
    fake_turns(monkeypatch, [TC("c1", "q")])
    fake_search(monkeypatch, exc=ValueError("Expecting value"))
    assert events(ask()) == [("error", {"message": app_module.RAG_DOWN})]

def test_ask_passes_history(monkeypatch, keyed):
    calls = fake_turns(monkeypatch, ["Yes."])
    history = [{"role": "user", "content": "What is MFA?"}, {"role": "assistant", "content": "A second factor."}]
    events(ask(history=history))
    assert calls[0]["messages"][1:-1] == history

def test_ask_llm_fails_mid_stream(monkeypatch, keyed):
    fake_turns(monkeypatch, ["Partial", RuntimeError("boom")])
    ev = events(ask())
    assert ev == [("delta", {"text": "Partial"}), ("error", {"message": app_module.LLM_FAILED})]
```
(Add `import json` at the top if it's missing.)

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_app.py -k ask -v`
Expected: the new tests FAIL, because `ask_events` still searches first.

- [ ] **Step 3: Implement `MAX_SEARCHES` and the new `ask_events` in `app.py`**

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/pytest -q`
Expected: all pass, with no skips left.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py && git commit -m "feat: /api/ask lets DeepSeek converse and search the knowledge base on demand"
```

---

### Task 3: Page, README and real-world check

**Files:**
- Modify: `static/app.js` (`renderSources`), `README.md`

**Interfaces:**
- Consumes: the SSE contract from Task 2, where `sources` may arrive 0–2 times.
- Produces:
  - `renderSources(container, sources)` appends rows (`container.append(...)`) instead of `replaceChildren`. Nothing else in the page changes.

- [ ] **Step 1: Change `renderSources` to append**

- [ ] **Step 2: Update `README.md`**

- What it is: it now chats normally. It searches your documents only for questions about them, and it says out loud when an answer is not from your documents.
- What leaves the machine: small talk and general questions send only the conversation. Document questions also send the matching passages.
- Troubleshooting: if the tunnel is down, small talk still works and document questions show the tunnel message.
- Performance: add the re-measured first-audio time for a document question, plus one small-talk time.

- [ ] **Step 3: Real check in the browser**

Restart the server: stop any running `app.py` on port 8765, then run `.venv/bin/python app.py`. The key is in `.env` and the tunnel is up. In Chrome (chrome-devtools MCP), type these in order and record each answer, whether sources appeared, and the `first audio after` value:
1. "hi". Expected: a friendly reply, no sources.
2. "explain zero trust". Expected: the answer starts by saying it is not from your documents, no sources.
3. "What does the access control policy say about MFA?" Expected: sources appear, and the answer is grounded in them.
4. "what about SAMA?" Expected: sources appear, and the search is about SAMA and MFA.

If DeepSeek skips the search for 3 or 4, or searches for 1 or 2, tune the `SYSTEM_PROMPT` or the tool description wording, then re-check. Stop the server afterwards, using its exact PID.

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/pytest -q && node --test tests/*.test.mjs`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add static/app.js README.md app.py && git commit -m "feat: append sources per search; document conversation mode"
```
