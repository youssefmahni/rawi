# Rawi

A demo voice assistant that hears English, Moroccan, Egyptian and Saudi Arabic and answers out loud
in the same language. Replies come from DeepSeek. By default speech-to-text (ElevenLabs Scribe) and
text-to-speech (one male ElevenLabs voice per language, set in `voices/voices.json`) are cloud
services; the local engines (faster-whisper and Piper) are still supported. Speak or type; the reply
streams to the page and is spoken sentence by sentence, with a badge showing the reply's dialect.

**What leaves the machine:** with the default settings, your recordings go to ElevenLabs
(speech-to-text), and every reply's text goes to DeepSeek (to write it) and to ElevenLabs (to speak
it). Set `STT_ENGINE=whisper` to transcribe locally and use Piper voices in `voices/voices.json` to
keep speech synthesis local; then only the question and reply text go to DeepSeek. The keys
(`DEEPSEEK_API_KEY`, `ELEVENLABS_API_KEY`) live in `.env`, which is never committed.

> The sections below were written for the local engines (Whisper and Piper). They still apply when
> you choose them.

## Prerequisites

- Python 3.14 (3.13 also works)
- ffmpeg
- A DeepSeek API key

## Setup (this machine)

```bash
python3 -m venv .venv
mkdir -p ~/.cache/pip-tmp && TMPDIR=~/.cache/pip-tmp .venv/bin/pip install -r requirements.txt
bash deploy/install.sh   # downloads the Piper voices and the speech-to-text model
cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY=sk-…
```

`TMPDIR` points pip at a roomier temp directory; without it, pip can fail with "No space left on
device" when `/tmp` is small. The first start downloads the Whisper `medium` model (~1.5 GB) to
`~/.cache/huggingface`.

Deployment to the mini PC (systemd user service, LAN access, mic over an SSH tunnel) is documented
in [`deploy/README.md`](deploy/README.md).

## Run

```bash
.venv/bin/python app.py
```

It prints "Loading Whisper medium…" and loads the speech models before serving, so the first
question isn't slowed by a model load. If a model fails to load, the app still starts and the speech
routes report the error.

Open http://127.0.0.1:8765 (or http://localhost:8765). The server answers only to the configured
host names (default `127.0.0.1,localhost`); any other `Host` gets 400, which blocks DNS-rebinding
attacks. It also rejects POSTs sent by other sites (403). LAN serving and HTTPS are opt-in via env
(see Config).

## Controls

- Tap the mic button to start recording, tap again to stop and send.
- Hold Space to talk, release to send (when no text field or button has focus).
- Esc stops the answer or the speech.
- Recordings shorter than 0.5 s show "Didn't catch that"; recording auto-stops after 60 s.
- Or type a question and press Enter.

## Config (env, in `.env`)

| Var | Default | Meaning |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | — | DeepSeek key (required) |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | override to simulate an outage |
| `STT_MODEL` | `medium` | faster-whisper model (`WHISPER_MODEL` still works as a fallback) |
| `STT_DEVICE` | `cpu` | `cpu` / `cuda` / `auto` |
| `STT_COMPUTE` | — | empty → `int8` on CPU, `float16` on CUDA |
| `VOICES_FILE` | `voices/voices.json` | per-dialect voice map |
| `PIPER_VOICE` | `en_US-lessac-medium` | fallback voice when `voices.json` is absent |
| `HOST` | `127.0.0.1` | bind address |
| `ALLOWED_HOSTS` | `127.0.0.1,localhost` | allowed Host names (comma list) |
| `SSL_CERTFILE` / `SSL_KEYFILE` | — | set both to serve HTTPS |
| `PORT` | `8765` | port |

## Performance

Measured on the mini PC (i5-6400T, 4 cores, no GPU): speech-to-text runs ~2× realtime (a 3 s
question ≈ 6 s to transcribe), and Piper synthesizes a sentence in ~0.5–0.9 s, so first audio lands
around 6–7 s for a spoken question. Full per-model numbers are in `bench/results/stt.md`; see
`bench/rehearsal.py` for the end-to-end measurement.

## Tests

```bash
.venv/bin/pytest -q
node --test tests/*.test.mjs
```

## Troubleshooting

| Message | Fix |
| --- | --- |
| DeepSeek API key missing. Add DEEPSEEK_API_KEY to .env and restart. | Add `DEEPSEEK_API_KEY=sk-...` to `.env` and restart. |
| DeepSeek request failed. Try again. | Network/API problem, invalid key, or no credit. Try again; check the key and balance. |
| I can't reach my reply service right now. … | The internet dropped mid-answer. The offline message is spoken and shown; check the connection. |
| Could not decode audio. | The recording could not be read. Make sure ffmpeg is installed, then retry. |
| Speech model unavailable. | The Whisper model failed to load. First run downloads ~1.5 GB; check network/disk and retry. |
| No voice for this dialect. | A dialect in `voices.json` has no voice. Check the voice map and that the voice files are in `voices/`. |
| Voice unavailable. | A Piper voice failed to load. Check `voices/` has the `.onnx` + `.onnx.json` files. |
| Speech synthesis failed. | Retry; check the voice files. |
| Speech request failed. | The page could not reach the server. Check that `app.py` is running. |
| Microphone access was denied. … / No microphone found. / Microphone unavailable. | Allow the mic in the browser site settings, connect a mic, or close apps holding it. |
| 400 Bad Request (Invalid host header) | Open the app at `http://127.0.0.1:8765` or `http://localhost:8765`, not another host name. |

Other notes:

- `requirements.txt` pins `av>=11,<17` because av 19 breaks faster-whisper 1.2.1.
- `.env` lines are `KEY=value`, not a bare key. An exported shell variable of the same name wins
  over `.env`.
