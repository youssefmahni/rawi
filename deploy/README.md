# Deploying Rawi to the mini PC

Rawi runs on the mini PC: speech-to-text (faster-whisper `medium`) and text-to-speech (Piper) stay
local, and only question/reply **text** goes to the DeepSeek API. The browser runs on a laptop or
tablet on the same network.

## 1. Install

```bash
bash ~/rawi/deploy/install.sh
```

Idempotent, no sudo. It creates `.venv`, installs `requirements.txt`, downloads the two Piper voices
(`en_US-lessac-medium`, `ar_JO-kareem-medium`) and the speech-to-text model, and copies
`voices.example.json` → `voices/voices.json` if absent.

## 2. The key

Create the key yourself (the install script never touches it):

```bash
nano ~/rawi/.env    # add: DEEPSEEK_API_KEY=sk-…
```

The app reads `.env` on start; `.env` is never copied, logged or sent to the page.

## 3. Run as a systemd **user** service

```bash
mkdir -p ~/.config/systemd/user
cp ~/rawi/deploy/rawi.service ~/.config/systemd/user/rawi.service
systemctl --user daemon-reload
systemctl --user enable --now rawi
```

To keep it running after logout/reboot, run once (needs sudo):

```bash
sudo loginctl enable-linger <user>
```

Check it: `systemctl --user status rawi`, and `curl -s http://127.0.0.1:8765/api/health`.

## 4. Reach it from the demo laptop/tablet

Browsers only expose the microphone on `https://` or `localhost`, so opening
`http://<minipc>:8765` on the laptop silently loses the mic. Use one of:

- **Recommended — SSH port-forward** (server stays on `127.0.0.1`, so the mic works):
  ```bash
  ssh -N -L 8765:127.0.0.1:8765 <user>@<minipc>
  ```
  then open `http://localhost:8765` on the laptop.

- **Tablet / HTTPS** — `mkcert` a cert for the mini PC hostname, then set in `.env`:
  `SSL_CERTFILE=…`, `SSL_KEYFILE=…`, `HOST=0.0.0.0`, `ALLOWED_HOSTS=<hostname>`, and install
  mkcert's root CA on the tablet. **Ask before binding `0.0.0.0`** — it exposes the app to the
  whole network.

## Config (env)

`STT_MODEL` (default `medium`), `STT_DEVICE` (`cpu`/`cuda`/`auto`), `STT_COMPUTE`,
`VOICES_FILE` (default `voices/voices.json`), `PIPER_VOICE`, `HOST`, `ALLOWED_HOSTS`,
`SSL_CERTFILE`, `SSL_KEYFILE`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `PORT`.
