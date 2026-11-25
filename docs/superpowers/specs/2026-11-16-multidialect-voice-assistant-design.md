# Multidialect Voice Assistant — Design

Date: 2026-11-16
Status: draft, awaiting approval
Replaces for this work: the knowledge-base parts of `2026-11-15-whisper-voice-assistant-design.md` and `2026-11-15-normal-conversation-design.md`
Schema page: https://claude.ai/artifact/E8md7NPwZX1wwZnhXLvs6d

## Intent

A live demo for clients in Saudi Arabia: a voice assistant people talk to in
English or in Moroccan, Egyptian or Saudi Arabic, and that answers out loud in
the same language and dialect. The "wow" moment is switching language or
dialect mid-conversation and hearing it switch with you. It must feel quick
and be reliable for a few minutes in front of people.

## Goals

- Understands English, Moroccan (Darija), Egyptian and Saudi Arabic speech.
- Replies in the language and dialect of the latest question, spoken in a voice for that dialect.
- Mixed Arabic + English questions get a reply in the Arabic dialect, with English terms kept as English words.
- Normal conversation only: greetings, small talk, general questions.
- Audio never leaves the mini PC: speech-to-text and text-to-speech run there on open models.
- First audio about 2–3 s after the speaker stops (target, measured on the mini PC).

## Non-goals

- The ciso-rag knowledge base, document search and sources (removed).
- Fully offline operation: the replies come from the DeepSeek API (user's decision, 2026-11-16).
- Other dialects (Levantine, Iraqi, …) or French mixed into Darija.
- Wake word, multi-user, accounts, mobile app.

## Anti-goals

- A reply in the wrong dialect, or a Moroccan question answered in a Saudi voice.
- Going silent: any failure (no internet, model error, unclear audio) is said out loud and shown.
- Long answers: replies stay 2–4 spoken sentences.

## Constraints

- Runs on a mini PC (hardware not yet known; checked over SSH before installing anything).
  The browser runs on a laptop or tablet on the same network.
- Speech-to-text and text-to-speech models must be open (downloadable, license allowing the demo).
  Licenses are checked during the tests, before a model is adopted.
- DeepSeek `deepseek-chat` via its API with the user's key, as the app does today. The key stays in `.env` and is never logged or sent to the page.
- The server only answers on the local network; same host and origin protections as today, adjusted from `127.0.0.1` to the mini PC's LAN address.

## Approaches

| | Approach | Trade-off |
|---|---|---|
| A | One general model per stage | Fewest models, least memory; Moroccan likely weak at every stage |
| B | Dialect specialists (MoulSot, DarijaTTS, …) chosen per dialect | Best per-dialect quality; 2–3× memory, realistically needs a GPU |
| **C (picked)** | General models first; a specialist only for the stage and dialect that fails a test | Load as low as the demo allows; every choice rests on a measurement |

Picked C (my recommendation; the user said "build decisions", accepting the proposals).
With DeepSeek writing the replies, the approach only concerns speech-to-text and voices.

## Decisions

Made by the user:
- Languages: English, Moroccan, Egyptian, Saudi; reply in the same dialect.
- Normal discussion only; no knowledge base.
- DeepSeek API as the language model.
- Demo for clients in Saudi Arabia; runs on a mini PC.

My calls (accepted via "build decisions"; each can be changed):
- **Setup:** browser on a laptop or tablet, mini PC on the same network.
- **Talking:** tap to talk; tapping again while it speaks interrupts it. Space and Esc keep working on a keyboard.
- **Topics:** polite, short; steers away from politics and religion ("I'm a demo assistant, let's talk about something else").
- **Persona:** a neutral name, placeholder "Rawi" (راوي). Voice gender follows what the chosen models offer; one gender across all four voices if possible (Lahgtna can copy a voice from a short sample).
- **No internet:** the assistant shows and says (with a recorded local clip in each language) that it can't reach its reply service.
- **Dialect detection:** DeepSeek labels its own reply with a dialect tag (`[en]`, `[ar-MA]`, `[ar-EG]`, `[ar-SA]`) as the first token; the server strips it, uses it to pick the voice and sends it to the page for the dialect badge. No extra model. If Darija transcription turns out poor, an audio dialect classifier is added in front of speech-to-text (approach C).
- **UI:** keep the visionOS look; Arabic messages right-to-left; a badge on each reply showing the dialect.
- **Code:** this repo, a new branch. `app.py` is split into `stt.py`, `llm.py`, `tts.py` and the routes, each model behind a small interface so a test result means swapping one model.

## Architecture

```
Browser (laptop/tablet)          Mini PC                                   Internet
mic → tap-to-talk → audio ──WS──► speech to text (general; Darija model if needed)
                                   │ transcript
                                   ├────────────── question + last turns ──► DeepSeek
                                   │ ◄──────────── "[ar-MA] reply…" streamed ─┘
                                   │ strip tag → dialect → voice
                                   │ sentence splitter (existing)
speaker ◄── audio per sentence ◄── text to speech: en → Piper
badge, transcript, reply ◄──────                   ar-EG / ar-SA → Lahgtna
                                                   ar-MA → DarijaTTS (or Lahgtna if it passes)
```

Speech-to-text candidates: Whisper large-v3 (faster-whisper), Jais-ASR (only if memory allows: 11.5 GB), MoulSot for Darija.
Voice candidates: Piper (English, kept), Lahgtna (Egyptian, Saudi, Moroccan claimed), DarijaTTS.

## Phases

1. **Tests that decide models** (before building the pipeline; results go to the user):
   - DeepSeek dialect samples: 5 questions per dialect; the user judges the replies.
   - Mini PC hardware check over SSH.
   - Speech-to-text: a few clips per dialect (EchoNova public set for Egyptian and Saudi; a Darija set for Moroccan) — error rate and time per clip on the mini PC.
   - Voices: the same sentences in each candidate voice; the user judges.
2. **Build** on the chosen models, with tests as today (pytest with models faked, node tests for the page logic).
3. **Deploy and rehearse** on the mini PC: measure first-audio per dialect, run the demo script end to end, with and without internet.

## Checks

- Each dialect: a rehearsed question gets a reply in that dialect and voice; the badge matches.
- Switching dialect between questions switches the voice.
- Mixed Arabic + English question → Arabic dialect reply.
- Internet off → spoken and shown message, no silence.
- First audio measured per dialect on the mini PC, reported against the 2–3 s target.

## Left to the builder

Prompt wording; how the tag is parsed from the stream; WebSocket vs keeping HTTP upload (WebSocket preferred if it saves time); model loading order and memory management; test structure.

## Open (blocks phase 1)

- The mini PC's SSH address (`user@host`).
- The demo date (decides how much of approach C's specialist swaps fit).
