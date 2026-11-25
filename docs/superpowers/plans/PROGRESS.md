# Progress log — multidialect voice assistant

Ruling: venv uses system Python 3.14, not 3.13 — faster-whisper/ctranslate2 (4.9.0), onnxruntime (1.32.0), piper-tts (1.8.0) and every other pin all have cp314 or abi3 wheels; the full requirements install and import cleanly on 3.14.

Ruling: installed Node 24.21.0 and ffmpeg 9.0.2 with `sudo apt install` (user provided the sudo password via the `pwd` file) — needed for the node test suite and later bench audio work.

## Task 1 — Branch and remove the knowledge base
- Branch `feat/multidialect-voice-assistant` created.
- Tests rewritten for the plain chat flow; pytest 34 pass, node 9 pass.
- Browser check deferred: no DeepSeek key yet (Task 4). Smoke-tested via curl — `/` 200, `/api/health` → `{"deepseek_key":false}`, `/api/ask` → `nokey` SSE error. Real spoken reply needs the key.
- Ruling: the plan's `grep -rn "rag\|classification\|sources" static/` still matches `localStorage`/`storage` (contains "rag"); no ciso-rag/sources/classification code remains — the `load`/`save` helpers and voice settings stay until Task 11.

## Task 2 — Split app.py into modules
- `config.py`, `dialects.py` (stub), `llm.py`, `stt.py`, `tts.py` created; `app.py` keeps only routes/middleware/lifespan and calls modules through module objects.
- Shared test helpers moved to `tests/conftest.py`; tests split into `test_llm.py`, `test_stt.py`, `test_tts.py`. `pytest.ini` now sets `pythonpath = . tests`.
- pytest 34 pass, node 9 pass (unchanged). Smoke run: `/` 200, `/api/health` ok, `/api/ask` → `nokey` SSE error.

## Task 3 — Dialect prompt, tag parser, dialect event, offline errors
- `dialects.py` filled in (`DIALECTS`, `DEFAULT_ARABIC`, `has_arabic`, `language_of`, `guess_dialect`); `llm.DialectTagParser`, `llm.OfflineError`, `llm.OFFLINE`, the full Rawi `SYSTEM_PROMPT`, and `config.DEEPSEEK_BASE_URL` added.
- `ask_events` now emits `dialect` before text, and `error` carries `kind` (`nokey`/`offline`/`failed`) + `language`; `AskBody.language` optional. `stream_turn` maps `openai.APIConnectionError` → `OfflineError`.
- pytest 54 pass, node 9 pass.

## Task 4 — DeepSeek dialect samples (gate G1)
- `bench/questions.json` (30 questions) and `bench/llm_dialects.py` written; `.gitignore` now ignores `bench/results/`.
- Ran twice (baseline + after adding few-shot example exchanges and an explicit "never tag Darija/Egyptian as ar-SA" rule): **tag match stayed 16/30**. DeepSeek writes each Arabic dialect correctly in the reply text but always emits `[ar-SA]`; English tags are 5/5 reliable.
- Gate G1: user judged the dialect text "all good" (content correct), so the problem is purely the self-label.
- Ruling: **route the Arabic voice by detecting the dialect from the question** (text/audio classifier), not from DeepSeek's reply tag. This adds a dialect-classifier component to be chosen in Task 8 and wired in Tasks 9/11. The tag is still stripped (never shown/spoken).

## Task 5 — Mini PC hardware check (gate G2)
- `bench/hardware.sh` written and run locally (we are on the mini PC). Result: Intel Core i5-6400T (4 cores / 4 threads, 2.2 GHz, AVX2 only, no AVX-512/AMX), 15 GB RAM (~13 GB available), Intel HD 530 (no NVIDIA/ROCm), 192 GB free disk, Python 3.14.10, ffmpeg 9.0.2, `systemctl --user` works. → **Light CPU** tier.
- Ruling: Jais-ASR skipped — needs ~14 GB free RAM, only 15 GB total / ~13 GB available.

## Task 6 — Speech-to-text benchmark (measurement, in progress)
- `bench/requirements-bench.txt`, `bench/prepare_stt_sets.py`, `bench/stt_bench.py` written; bench venv at `~/rawi-bench/stt/.venv` (datasets 5.2, faster-whisper 1.2.1, arabic-asr-benchmark 1.0.0).
- Ruling: echonova Egyptian/Saudi and DODa Moroccan are **gated** (403 "not in authorized list" even after `hf auth login` as ysf-p4); only public READMEs visible. Egyptian/Saudi measurement deferred (will use published echonova WER for large-v3 in Task 8, or a non-gated source).
- Ruling: Moroccan set switched to `FatimahEmadEldin/Moroccan-Arabic-Multimodal-Emotion-Recognition` (MDER-MA, non-gated; Darija audio+transcripts, 48 kHz stereo, "other" license via Mendeley DOI) instead of DODa.

## Task 7 — Voice samples (measurement, in progress)
- `bench/tts_sentences.json`, `bench/tts_samples.py`, `bench/make_listen.py` written.
- Piper baseline working on the mini PC: `en_US-lessac-medium` (English) and `ar_JO-kareem-medium` (Arabic, MSA/Jordanian) downloaded to `voices/`; synth ~0.5–1.8 s per ~5 s clip (RTF ~0.15). Samples generated for all four dialects; `bench/results/tts/listen.html` written.
- TTS candidates researched (for Task 8):
  - `oddadmix/lahgtna-chatterbox-v0` — MIT, Chatterbox (ResembleAI) based (~3B), covers ar-SA/ar-EG/ar-MA, voice cloning via audio prompt; newer `v1` exists; heavy on CPU.
  - `KandirResearch/DarijaTTS-v0.1-500M` — Apache-2.0, 500M (OuteTTS), GGUF Q8_0 via `outetts==0.3.3` + `llama-cpp-python`; Moroccan only; lighter.
- Piper ar_JO is MSA/Jordanian only — not dialect-specific (the "MSA fallback"); dialect voices need Lahgtna and/or DarijaTTS (decision at G3/G4).

## Task 6 — results so far
- `large-v3-turbo` (int8, CPU): **RTF ≈ 3.3×** (median ~16.7 s per ~5–7 s clip); English WER **2.63%**; Darija language detection mislabels 7/30 clips as non-`ar` (en/ko/it/pt).
- Ruling: turbo is far too slow for the 2–3 s target, so I'm also benchmarking `small` and `medium` (multilingual) to find the fastest workable model (user approved). `large-v3` and `small` are running; `medium` next.

## Task 6 — complete (measurement + stt.md)
- Four models benchmarked (English LibriSpeech + Moroccan MDER-MA; Egyptian/Saudi gated→403, deferred to published WER):

| Model | Eng WER | RTF en | RTF Darija | Darija non-ar | Hangs>30s |
|---|---|---|---|---|---|
| large-v3 | 2.13% | 4.9× | 4.2× | 6/30 | 7 |
| large-v3-turbo | 2.63% | 3.3× | 3.4× | 7/30 | 0 |
| medium | 4.11% | 1.9× | 2.1× | 7/30 | 0 |
| small | 3.12% | 1.3× | 1.9× | 6/30 | 4 |

- Key: 2–3 s target unreachable (fastest `small` ≈ 1.3× realtime → ~4 s STT). `turbo`/`medium` never hallucinate on Darija; `small` hangs on 4/30 clips. `medium` = speed/stability middle. All models mislabel ~20% of Darija → en/ar retry mandatory. `stt.md` written; scripts committed.

## Task 7 — gate G3 (voices decided)
- Ruling (G3): **Piper-only voices** — `en` → `en_US-lessac-medium`, all Arabic → `ar_JO-kareem-medium` (Jordanian/MSA). No dialect-specific engines: DarijaTTS is blocked (llama-cpp-python has no Py3.14 wheels), and Lahgtna (3B) is a heavy torch install with slow CPU inference.
- Consequence: the Arabic voice no longer switches per dialect, so the voice only needs en vs ar (from `has_arabic`); sub-dialect detection (the G1 classifier) is no longer needed for the voice, only for the cosmetic badge.

## Tasks 9–14 (build + deploy)
- Task 9: multilingual STT — `stt.Transcript`/`Transcriber`/`get_transcriber`, en/ar detection with re-transcribe fallback (ties → ar), `STT_MODEL/STT_DEVICE/STT_COMPUTE` in config, `/api/transcribe` returns `{text, language}`.
- Task 10: voice per dialect — `tts.load_voice_map`/`get_engine`/`synthesize(text, dialect)`/`available_dialects`, `/api/speak` by dialect, health lists voices, `/api/voices` removed.
- Task 11: page — RTL, dialect badge, per-dialect voice, offline message spoken; removed the Settings sheet (dialog, gear, Ctrl/Cmd+,, voice localStorage); `voiceLabel` removed.
- Task 12: LAN/HTTPS/timing (done earlier).
- Task 13: `deploy/install.sh` (idempotent, no sudo), `deploy/rawi.service` (systemd user unit), `deploy/README.md`; ran install locally (we are on the mini PC); `voices/voices.json` created from `voices.example.json`.
- Task 14: `bench/rehearsal.py`, `docs/demo-runbook.md`, `README.md` rewritten for Rawi; `.gitignore` adds `bench/demo_clips/`.

Test counts along the way: pytest 54 → 58 (Task 12) → 61 (Task 9) → 62 (Task 10); node 15 → 14 (voiceLabel removed).




## Early Task 11 (frontend, done while benchmarks run)
- `static/dialects.js` (DIALECTS/hasArabic/languageOf) + `tests/dialects.test.mjs`; `SentenceBuffer` now splits on `؟` and treats non-Latin letters after `.` as a sentence end. node 15 pass, pytest 54 pass.

## Task 12 — LAN access, HTTPS option, timing logs (done early; no gate)
- `config` now reads `HOST`, `ALLOWED_HOSTS`, `SSL_CERTFILE`/`SSL_KEYFILE`, plus `allowed_origins()` and `uvicorn_ssl_kwargs()`; `SameOriginMiddleware`/`TrustedHostMiddleware` use them; `uvicorn.run` honors host + TLS. `timing stt_ms=`/`llm_first_ms=` INFO logs added. `.env.example` documents the new vars. pytest 58 pass, node 15 pass.








