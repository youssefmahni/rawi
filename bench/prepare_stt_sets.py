"""Prepare the STT evaluation sets (30 clips each) into bench/results/stt/.

For each set, writes:
  - bench/results/stt/refs_<set>.jsonl  ({"id","dialect","recording_id","text"})
  - bench/results/stt/audio/<set>/<id>.wav

Licenses (noted for the decision record):
  - egyptian / saudi: echonova-ai/arabic-dialect-asr-benchmark (gated; transcripts CC BY 4.0,
    audio research/evaluation only, no redistribution).
  - moroccan: atlasia/DODa-audio-dataset (MIT; transcripts human-authored then Wit.ai-corrected).
  - english: openslr/librispeech_asr (test.clean; CC BY 4.0).
"""

import json
import sys
from pathlib import Path

BENCH = Path(__file__).parent
OUT = BENCH / "results" / "stt"
AUDIO = OUT / "audio"

N = 30

# (set, dataset_args, split, dialect_filter, text_key, id_prefix)
SETS = {
    "egyptian": ("echonova-ai/arabic-dialect-asr-benchmark", "test", "egyptian", "text", "egyptian"),
    "saudi": ("echonova-ai/arabic-dialect-asr-benchmark", "test", "saudi", "text", "saudi"),
    "moroccan": ("FatimahEmadEldin/Moroccan-Arabic-Multimodal-Emotion-Recognition", "test", None, "transcription", "moroccan"),
    "english": ("openslr/librispeech_asr", "test", None, "text", "english"),
}


def load_rows(repo, split, config=None):
    from datasets import load_dataset

    if config:
        return load_dataset(repo, name=config, split=split, streaming=True)
    return load_dataset(repo, split=split, streaming=True)


def write_wav(path, array, sampling_rate):
    import soundfile as sf

    sf.write(str(path), array, sampling_rate, subtype="PCM_16")


def prepare():
    AUDIO.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    only = set(sys.argv[1:])
    for setname, (repo, split, dialect, text_key, id_prefix) in SETS.items():
        if only and setname not in only:
            continue
        config = "clean" if setname == "english" else None
        rows = load_rows(repo, split, config)
        (AUDIO / setname).mkdir(parents=True, exist_ok=True)
        refs = []
        seen = 0
        for row in rows:
            if dialect is not None and row.get("dialect") != dialect:
                continue
            text = row.get(text_key)
            if not text or len(str(text).strip()) < 4:
                continue
            rid = row.get("id") or row.get("file") or row.get("file_name")
            cid = f"{id_prefix}_{seen + 1:05d}"
            dest = AUDIO / setname / f"{cid}.wav"
            audio = row.get("audio")
            if audio is not None:
                write_wav(dest, audio["array"], audio["sampling_rate"])
            elif row.get("file_name"):
                from huggingface_hub import hf_hub_download
                import soundfile as sf

                src = hf_hub_download(repo, row["file_name"], repo_type="dataset")
                array, sr = sf.read(src, dtype="float32")
                write_wav(dest, array, sr)
            else:
                continue
            refs.append({
                "id": cid,
                "dialect": "english" if setname == "english" else (dialect or "moroccan"),
                "recording_id": str(rid) if rid is not None else cid,
                "text": str(text).strip(),
            })
            seen += 1
            if seen >= N:
                break
        (OUT / f"refs_{setname}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in refs) + "\n"
        )
        print(f"{setname}: {len(refs)} clips")


if __name__ == "__main__":
    prepare()
