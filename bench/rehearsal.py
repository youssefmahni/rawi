"""Rehearsal harness: measure end-to-end first-audio per demo clip.

Usage: python bench/rehearsal.py [base_url]   (default http://127.0.0.1:8765)

Reads bench/demo_clips/<n>_<dialect>.<ext> in filename order, keeps history across clips, and for
each clip:
  1. POST /api/transcribe  → stt_ms, transcript, detected language
  2. POST /api/ask         → llm_first_ms (time to first delta), dialect tag, reply text
  3. POST /api/speak       → tts_first_ms (first sentence, split like speech.js)

Writes bench/results/rehearsal.md (one row per clip, medians per dialect, tag-match pass mark).
"""

import json
import re
import sys
import time
from pathlib import Path

import httpx

BENCH = Path(__file__).parent
CLIPS = BENCH / "demo_clips"
RESULTS = BENCH / "results"


def sentences(text):
    # Same sentence boundary as static/speech.js for the common case: [.!?؟] + whitespace.
    return [s.strip() for s in re.split(r"[.!?؟]\s+", text) if s.strip()]


def sse_events(resp):
    buf = ""
    for chunk in resp.iter_text():
        buf += chunk
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            event, data = None, None
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
            if data is not None:
                yield event, json.loads(data)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    clips = sorted(CLIPS.glob("*"))
    if not clips:
        raise SystemExit(f"no clips in {CLIPS} — record demo questions there first (see demo-runbook)")
    history = []
    rows = []
    with httpx.Client(base_url=base, timeout=None) as client:
        for clip in clips:
            expected = clip.stem.split("_", 1)[1] if "_" in clip.stem else "?"

            t0 = time.perf_counter()
            tr = client.post("/api/transcribe", content=clip.read_bytes())
            stt_ms = (time.perf_counter() - t0) * 1000
            tr = tr.json()
            text, lang = tr.get("text", ""), tr.get("language", "en")

            t0 = time.perf_counter()
            llm_first_ms = None
            dialect = None
            reply = ""
            with client.stream("POST", "/api/ask", json={"question": text, "history": history}) as resp:
                for event, data in sse_events(resp):
                    if event == "dialect":
                        dialect = data.get("dialect")
                    elif event == "delta":
                        reply += data.get("text", "")
                        if llm_first_ms is None:
                            llm_first_ms = (time.perf_counter() - t0) * 1000
                    elif event == "error":
                        break

            first = sentences(reply)[0] if sentences(reply) else ""
            tts_first_ms = None
            if first and dialect:
                t0 = time.perf_counter()
                client.post("/api/speak", json={"text": first, "dialect": dialect})
                tts_first_ms = (time.perf_counter() - t0) * 1000

            first_audio = (stt_ms + (llm_first_ms or 0) + (tts_first_ms or 0)) / 1000
            rows.append(dict(
                clip=clip.stem, expected=expected, transcript=text, language=lang, dialect=dialect,
                stt_ms=round(stt_ms), llm_first_ms=round(llm_first_ms) if llm_first_ms is not None else None,
                tts_first_ms=round(tts_first_ms) if tts_first_ms is not None else None,
                first_audio_s=round(first_audio, 2), reply=reply))
            history = (history + [
                {"role": "user", "content": text}, {"role": "assistant", "content": reply}])[-6:]
            print(f"{clip.stem}: stt={stt_ms:.0f}ms llm_first={llm_first_ms and round(llm_first_ms)}ms "
                  f"tts_first={tts_first_ms and round(tts_first_ms)}ms dialect={dialect}")

    RESULTS.mkdir(exist_ok=True)
    out = ["# Rehearsal measurements\n", "| Clip | Expected | Detected lang | Dialect tag | STT ms | LLM first ms | TTS first ms | First audio (s) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tag = r["dialect"] or "—"
        ok = "✓" if r["dialect"] == r["expected"] else "✗"
        out.append(f"| {r['clip']} | {r['expected']} | {r['language']} | {tag} {ok} | {r['stt_ms']} | "
                   f"{r['llm_first_ms'] or '—'} | {r['tts_first_ms'] or '—'} | {r['first_audio_s']} |")
    matches = sum(1 for r in rows if r["dialect"] == r["expected"])
    out.append(f"\nTag match: {matches}/{len(rows)}")
    # medians per dialect
    by = {}
    for r in rows:
        by.setdefault(r["expected"], []).append(r["first_audio_s"])
    out.append("\n| Dialect | Median first audio (s) |")
    out.append("|---|---|")
    for d in sorted(by):
        med = sorted(by[d])[len(by[d]) // 2]
        out.append(f"| {d} | {med:.2f} |")
    (RESULTS / "rehearsal.md").write_text("\n".join(out) + "\n")
    print(f"wrote {RESULTS / 'rehearsal.md'} — tag match {matches}/{len(rows)}")


if __name__ == "__main__":
    main()
