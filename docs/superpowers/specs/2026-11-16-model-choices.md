# Model choices — Rawi (2026-11-16)

Decisions for the multidialect voice assistant, from Tasks 4–7 measurements and gates G1–G4.

## Summary

| Stage | Pick | Device | License |
|---|---|---|---|
| Speech-to-text | faster-whisper **`medium`** (int8, beam_size=1, vad_filter=True, language auto + en/ar retry) | CPU | MIT (faster-whisper/CTranslate2), MIT (Whisper weights) |
| Voice (English) | Piper `en_US-lessac-medium` | CPU | Piper (MIT code; voice from rhasspy/piper-voices) |
| Voice (Arabic) | Piper `ar_JO-kareem-medium` (Jordanian/MSA) for ar-MA, ar-EG, ar-SA | CPU | Piper (MIT code; voice from rhasspy/piper-voices) |
| Reply LLM | DeepSeek `deepseek-chat` (unchanged) | API | — |

## Speech-to-text

Four models measured on the mini PC (i5-6400T, 4 cores, 15 GB, no GPU): 30 English
(LibriSpeech `test.clean`) + 30 Moroccan (MDER-MA). Egyptian/Saudi not measured locally —
the plan's source (`echonova-ai/arabic-dialect-asr-benchmark`) is gated (403 after login);
Arabic accuracy reference is the published EchoNova WER for Whisper large-v3
(Egyptian 14.85 %, Saudi 20.61 %).

| Model | Eng WER | RTF en | RTF Darija | Darija non-ar | Hangs >30 s |
|---|---|---|---|---|---|
| large-v3 | 2.13 % | 4.9× | 4.2× | 6/30 | 7 |
| large-v3-turbo | 2.63 % | 3.3× | 3.4× | 7/30 | 0 |
| **medium** | 4.11 % | 1.9× | 2.1× | 7/30 | **0** |
| small | 3.12 % | 1.3× | 1.9× | 6/30 | 4 |

- **Pick: `medium`** — the fastest model that never hallucinated on Darija (0 hangs), at ~2×
  realtime (~6–7 s first audio for a 3 s question). The 2–3 s target is unreachable on this CPU.
  Runner-up: `small` (1.3×, but 4/30 Darija clips looped 65–156 s). `large-v3`/`-turbo` are too slow.
- All models mislabel ~20 % of Darija clips as another language (en/ko/it/pt/fr/…), so the
  en/ar re-transcribe fallback is mandatory (Task 9).

## Voices

- Piper `en_US-lessac-medium` (English) and `ar_JO-kareem-medium` (Arabic) both run fast on CPU
  (RTF ~0.15; ~0.5–0.9 s per sentence).
- **Pick: Piper-only** (G3). `ar_JO-kareem-medium` is Jordanian/MSA, so all three Arabic dialects
  share one voice — the reply text switches dialect but the voice does not. Dialect-specific engines
  were rejected: DarijaTTS is blocked (llama-cpp-python has no Python 3.14 wheels), and Lahgtna
  Chatterbox (3B) needs a heavy torch install and slow CPU inference.
- `voices.example.json` (committed) maps `en → en_US-lessac-medium`, `ar-* → ar_JO-kareem-medium`;
  the real `voices/voices.json` is created on the mini PC (Task 13).

## Routing & detection

- Voice needs only en vs ar (from `has_arabic`); the earlier G1 idea of a question-level sub-dialect
  classifier is no longer needed for the voice — only the badge could use it, and it's cosmetic.
- Darija is handled by `medium` directly (no specialist, `DARIJA_ROUTING=none`).

## Rulings (gates and deviations)

- G1: reply content judged correct; DeepSeek self-tags all Arabic `[ar-SA]` (16/30 tags), so the
  tag is stripped but not trusted for routing.
- G2: Light CPU tier accepted (4 cores / 15 GB / no GPU).
- G3: Piper-only voices (one Arabic voice).
- G4: STT = `medium`.
- Jais-ASR skipped (needs ~14 GB RAM; only ~13 GB available).
- Egyptian/Saudi eval gated → deferred to published WER.
