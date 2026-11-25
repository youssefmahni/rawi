"""Generate TTS samples for a dialect+engine. Usage:

    python bench/tts_samples.py <engine> <dialect>

Engines: piper (en -> en_US-lessac-medium; ar dialects -> ar_JO-kareem-medium),
plus dialect engines as they are installed (chatterbox, outetts).

Writes bench/results/tts/<dialect>/<engine>__<n>.wav and a JSONL timing log.
"""

import argparse
import io
import json
import sys
import time
import wave
from pathlib import Path

BENCH = Path(__file__).parent
sys.path.insert(0, str(BENCH.parent))
RESULTS = BENCH / "results" / "tts"

PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "ar-SA": "ar_JO-kareem-medium",
    "ar-EG": "ar_JO-kareem-medium",
    "ar-MA": "ar_JO-kareem-medium",
}


def synth_piper(text, voice):
    from piper import PiperVoice

    model = PiperVoice.load(f"voices/{voice}.onnx")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        model.synthesize_wav(text, w)
    return buf.getvalue()


SYNTH = {"piper": synth_piper}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("dialect")
    args = ap.parse_args()

    sentences = json.loads((BENCH / "tts_sentences.json").read_text())[args.dialect]
    outdir = RESULTS / args.dialect
    outdir.mkdir(parents=True, exist_ok=True)

    if args.engine == "piper":
        voice = PIPER_VOICES[args.dialect]
        synth = lambda text: SYNTH["piper"](text, voice)  # noqa: E731
    elif args.engine in SYNTH:
        synth = lambda text: SYNTH[args.engine](text)  # noqa: E731
    else:
        raise SystemExit(f"unknown engine {args.engine}")

    # warm-up on the first sentence, then time each
    rows = []
    for i, s in enumerate(sentences):
        synth(s) if i == 0 else None  # warm
        t0 = time.perf_counter()
        data = synth(s)
        dt = time.perf_counter() - t0
        path = outdir / f"{args.engine}__{i}.wav"
        path.write_bytes(data)
        with wave.open(io.BytesIO(data)) as w:
            sr, dur = w.getframerate(), w.getnframes() / w.getframerate()
        rows.append({"engine": args.engine, "dialect": args.dialect, "n": i,
                     "seconds": round(dt, 3), "audio_seconds": round(dur, 2),
                     "sample_rate": sr, "text": s})
        print(f"{args.dialect}/{args.engine}__{i}: {dt:.2f}s -> {dur:.2f}s audio")
    (RESULTS / f"times_{args.engine}_{args.dialect}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )


if __name__ == "__main__":
    main()
