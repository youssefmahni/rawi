# Rawi — demo runbook

## The day before

1. Sync the code to the mini PC and re-run the install (it's idempotent):
   ```bash
   cd ~/rawi && git pull && bash deploy/install.sh
   ```
2. Start the app and run the rehearsal harness (after recording demo clips into
   `bench/demo_clips/<n>_<dialect>.<ext>` — the filename carries the expected dialect):
   ```bash
   systemctl --user restart rawi     # or: .venv/bin/python app.py &
   .venv/bin/python bench/rehearsal.py
   cat bench/results/rehearsal.md
   ```
   Confirm every clip's dialect tag matches its expected dialect, and note the median first-audio
   per dialect.
3. Charge the laptop and confirm the mic/headset work.

## At the venue

1. **Internet check** — the replies need DeepSeek:
   ```bash
   curl -s https://api.deepseek.com >/dev/null && echo online
   ```
2. **Service check**:
   ```bash
   systemctl --user status rawi
   curl -s http://127.0.0.1:8765/api/health   # expect deepseek_key:true and all voices true
   ```
3. **Start the tunnel** (from the demo laptop — keeps the mic working):
   ```bash
   ssh -N -L 8765:127.0.0.1:8765 <user>@<minipc>
   ```
4. **Open** `http://localhost:8765` on the laptop.
5. **One test sentence per dialect** (type them): English, Moroccan, Egyptian, Saudi — confirm the
   badge, direction and voice follow.
6. **Headset/mic placement** — put the mic where the speaker will stand; test one spoken question.

## If something breaks

- **No reply / "can't reach my reply service"** — internet is down; the spoken+shown offline message
  is expected. Check the connection and retry.
- **Wrong dialect on the badge** — ask again with a longer sentence (DeepSeek's tag is unreliable for
  Arabic; the reply text still follows the question).
- **Silence (no audio)** — check the status pill: "Voice missing: …" means a voice file is missing
  (re-run `bash deploy/install.sh`); "Server unreachable" means the app/tunnel is down.
- **Mic not working on the laptop** — you're on `http://<minipc>:8765` instead of the tunnel's
  `http://localhost:8765`; browsers only allow the mic on localhost/https.

## The demo script (switch sequence)

Ask, in order:

1. "Hi, who are you?" *(English)*
2. "لاباس عليك؟" *(Moroccan Darija)*
3. "طب إنت منين؟" *(Egyptian)*
4. "وش تنصحني أسوي اليوم؟" *(Saudi)*
5. "Thanks, bye!" *(English)*

The "wow" is the language/dialect switching with each turn and the reply coming back in the same
language.
