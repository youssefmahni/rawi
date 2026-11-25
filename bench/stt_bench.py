"""Speech-to-text benchmark for the Rawi mini PC.

Usage:  python bench/stt_bench.py <model-id>

model-id is a faster-whisper model:  large-v3-turbo | large-v3
(light CPU: int8, beam_size=1, vad_filter=True, language=None auto-detected).

For each prepared clip (see prepare_stt_sets.py) it records the transcript,
wall time, and detected language, and writes:
  - bench/results/stt/<model>.jsonl        combined predictions (id, prediction)
  - bench/results/stt/<model>_<set>.jsonl  per-set predictions (for the scorer)
  - bench/results/stt/<model>.meta.jsonl   per-clip seconds + detected language
It also computes English WER inline. Arabic WER is scored separately with
`asr-bench evaluate` (echonova-v1 normalizer).
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

BENCH = Path(__file__).parent
sys.path.insert(0, str(BENCH.parent))

OUT = BENCH / "results" / "stt"
AUDIO = OUT / "audio"
SETS = ["egyptian", "saudi", "moroccan", "english"]


def load_refs(setname):
    return [json.loads(l) for l in (OUT / f"refs_{setname}.jsonl").read_text().splitlines() if l.strip()]


def load_model(model_id):
    from faster_whisper import WhisperModel

    return WhisperModel(model_id, device="cpu", compute_type="int8", cpu_threads=4)


def transcribe(model, path):
    segments, info = model.transcribe(str(path), language=None, vad_filter=True, beam_size=1)
    text = "".join(seg.text for seg in segments).strip()
    return text, info.language


def english_wer(refs, preds):
    import editdistance

    def norm(s):
        return [w for w in re.sub(r"[^\w\s]", " ", s.lower()).split()]

    err = tot = 0
    for r, p in zip(refs, preds):
        rw, pw = norm(r["text"]), norm(p["prediction"])
        err += editdistance.eval(rw, pw)
        tot += len(rw)
    return 100.0 * err / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    args = ap.parse_args()
    model_id = args.model_id

    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(model_id)

    # only benchmark sets whose refs are prepared
    sets = [s for s in SETS if (OUT / f"refs_{s}.jsonl").exists()]
    if not sets:
        raise SystemExit("no prepared refs — run prepare_stt_sets.py first")

    # warm-up: one throwaway clip
    warm = load_refs(sets[0])[0]
    transcribe(model, AUDIO / sets[0] / f"{warm['id']}.wav")

    all_preds = {}
    meta = []
    for setname in sets:
        refs = load_refs(setname)
        preds = []
        for r in refs:
            path = AUDIO / setname / f"{r['id']}.wav"
            t0 = time.perf_counter()
            text, lang = transcribe(model, path)
            secs = time.perf_counter() - t0
            preds.append({"id": r["id"], "prediction": text})
            all_preds[r["id"]] = text
            meta.append({"id": r["id"], "set": setname, "seconds": round(secs, 3),
                         "language": lang, "text": text})
            print(f"{setname}/{r['id']}: {secs:.2f}s lang={lang}")
        (OUT / f"{model_id}_{setname}.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in preds) + "\n"
        )
        if setname == "english":
            print(f"english WER ({model_id}): {english_wer(refs, preds):.2f}%")

    (OUT / f"{model_id}.jsonl").write_text(
        "\n".join(json.dumps({"id": k, "prediction": v}, ensure_ascii=False)
                  for k, v in sorted(all_preds.items())) + "\n"
    )
    (OUT / f"{model_id}.meta.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in meta) + "\n"
    )
    print(f"wrote {OUT}/{model_id}.jsonl and {model_id}.meta.jsonl")


if __name__ == "__main__":
    main()
