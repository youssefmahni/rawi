"""Generate bench/results/tts/listen.html — one <audio> per sample, grouped by dialect."""

import json
from pathlib import Path

BENCH = Path(__file__).parent
TTL = BENCH / "results" / "tts"
DIALECTS = ["en", "ar-MA", "ar-EG", "ar-SA"]

LABELS = {"en": "English", "ar-MA": "Moroccan", "ar-EG": "Egyptian", "ar-SA": "Saudi"}


def main():
    TTL.mkdir(parents=True, exist_ok=True)
    parts = ["<!doctype html><meta charset='utf-8'><title>TTS samples</title>",
             "<style>body{font-family:system-ui;background:#141824;color:#fff;padding:24px}"
             "h2{margin-top:32px}div{margin:10px 0}audio{width:100%;max-width:480px}"
             "p{color:#aaa;font-size:14px}</style>"]
    for d in DIALECTS:
        parts.append(f"<h2>{LABELS[d]} ({d})</h2>")
        wavs = sorted((TTL / d).glob("*.wav")) if (TTL / d).exists() else []
        if not wavs:
            parts.append("<p>no samples</p>")
            continue
        for w in wavs:
            parts.append(f"<div><p>{w.name}</p><audio controls src='{d}/{w.name}'></audio></div>")
    (TTL / "listen.html").write_text("\n".join(parts))
    print(f"wrote {TTL / 'listen.html'}")


if __name__ == "__main__":
    main()
