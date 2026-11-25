# Normal Conversation — Design (small change)

Date: 2026-11-15
Status: draft, awaiting approval
Builds on: `2026-11-15-whisper-voice-assistant-design.md`

## Intent

The assistant should also hold a normal conversation — greetings, small talk,
follow-ups, and general questions — not only answer from the CISO knowledge
base. The user chose **full assistant**: it answers anything; when an answer
does not come from the user's documents it says so out loud and shows no
sources.

## Goals

- Small talk ("hi", "thanks", "who are you?") gets a natural reply, no search, no sources.
- General questions ("explain zero trust") get an answer from DeepSeek's own
  knowledge, flagged in speech, e.g. "That's not in your documents, but generally…".
- Questions about the user's documents (policies, NCA ECC, SAMA CSF, PDPL,
  architecture, projects, Apple HIG) still search ciso-rag and answer from the passages, with sources shown.
- Follow-ups like "what about SAMA?" search with a full, self-contained query.

## Anti-goals

- Small talk or general questions sending document passages to DeepSeek.
- An answer drawn from general knowledge that sounds like it came from the user's documents.

## Changed from the original design

- The original anti-goal "answers that invent content not in the retrieved
  passages" is narrowed to: *a knowledge-base answer must stay within its
  passages, and a non-document answer must say so.*
- The non-goal "query rewriting for follow-ups" is removed — DeepSeek now writes the search query.
- The fixed no-results reply ("I couldn't find that in the knowledge base."
  without calling DeepSeek) is removed; DeepSeek answers instead, flagged.

## Approaches

| | Approach | Trade-off |
|---|---|---|
| **A (picked)** | DeepSeek gets a `search_knowledge_base` tool and decides when to call it | Small talk sends no passages; fixes follow-ups; +0.5–1.5 s before speech on document questions (one extra DeepSeek round trip) |
| B | Always search; prompt says "use passages if relevant, else chat" | Smallest change, but every "hi" sends 5 possibly-confidential passages and shows unrelated sources |
| — | Search always, keep only high-scoring passages | Dropped: rerank scores aren't calibrated (a weak match scored 0.999) |

User chose A.

## Design

**Tool** (OpenAI-style function calling; DeepSeek `deepseek-chat` supports it):
`search_knowledge_base(query: string)` — "Search the user's CISO knowledge base:
internal policies, NCA ECC, SAMA CSF, PDPL, ISO 27001, NIST CSF, estate
architecture, project plans, Apple HIG. Use a complete, self-contained query."

**Server flow for `/api/ask`** (replaces the search-first flow):
1. Key check as today (`NO_KEY`).
2. Call DeepSeek (streaming) with system prompt + last 6 history messages +
   the question, and the tool available.
3. If it streams text → forward as `delta` events, then `done`. No search, no `sources` event.
4. If it calls the tool → run `search_rag(query, body.classification)` (the
   classification setting still applies, fail-closed), send a `sources`
   event with the passages, give DeepSeek the passages as the tool result
   (or "No matching passages found." when empty), then stream its answer as `delta` + `done`.
5. At most **2** searches per answer; after that the tool is not offered again.
6. Errors: ciso-rag unreachable during a search → `error` RAG_DOWN (as today);
   DeepSeek failure → `error` LLM_FAILED (as today).

**SSE contract change:** `sources` is now optional — sent only when a search
ran (and may come more than once if two searches ran; the page appends).
`delta`/`done`/`error` unchanged. History stays text-only (tool calls/results
are not stored in history).

**System prompt** (builder may tune wording), covering: friendly spoken voice
assistant for a CISO; use the tool for anything about the user's documents
and regulations; when answering from passages, stay within them; when
answering from general knowledge, start by saying it's not from their
documents; 2–4 short spoken sentences; no markdown, lists, document names or passage numbers.

**Page:** handles answers without a `sources` event (it already does) and
appends sources if the event arrives more than once. No other UI change.

## Checks

- Tests (DeepSeek and ciso-rag faked): no tool call → no search and no
  `sources`; tool call → search gets the model's query and the classification,
  `sources` sent, second DeepSeek call gets the tool result; empty search →
  tool result says none found and the model still answers; RAG down during a
  search → `error` RAG_DOWN; search cap of 2.
- Manual: "hi", "explain zero trust", "what does the access control policy say
  about MFA?", then the follow-up "what about SAMA?" — check sources appear only
  for the document questions and the general answer is flagged. Re-measure
  first-audio for a document question; README updated.

## Left to the builder

Exact prompt and tool-description wording; how streamed tool-call fragments
are accumulated; test structure.
