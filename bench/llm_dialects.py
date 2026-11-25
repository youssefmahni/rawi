"""Run the DeepSeek dialect samples and write bench/results/llm_dialects.md.

Reads the key only through `config` (env/.env); never prints it.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

BENCH = Path(__file__).parent
sys.path.insert(0, str(BENCH.parent))  # repo root: config, llm, dialects

import config  # noqa: E402,F401  (loads .env)
import dialects  # noqa: E402
import llm  # noqa: E402

RESULTS = BENCH / "results"
RESULTS.mkdir(exist_ok=True)

QUESTIONS = json.loads((BENCH / "questions.json").read_text())

EXPECTED = {
    "en": ["en"] * 5,
    "ar-MA": ["ar-MA"] * 5,
    "ar-EG": ["ar-EG"] * 5,
    "ar-SA": ["ar-SA"] * 5,
    "mixed": ["ar-SA", "ar-MA", "ar-EG"],
    "switch": ["en", "ar-MA", "ar-EG", "ar-SA", "en"],
    "guard": ["en", "ar-SA"],
}


async def one_question(question: str, history: list[dict]):
    messages = llm.build_messages(question, history)
    parser = llm.DialectTagParser()
    text = []
    first_ms = None
    start = time.perf_counter()
    async for chunk in llm.stream_turn(messages):
        tag, t = parser.feed(chunk)
        if t:
            if first_ms is None:
                first_ms = (time.perf_counter() - start) * 1000
            text.append(t)
    tag, t = parser.finish()
    if t:
        if first_ms is None:
            first_ms = (time.perf_counter() - start) * 1000
        text.append(t)
    reply = "".join(text)
    dialect = tag or dialects.guess_dialect(reply)
    return dialect, reply, first_ms


async def main():
    rows = []
    match = total = 0
    for setname, questions in QUESTIONS.items():
        history = []
        for i, q in enumerate(questions):
            expected = EXPECTED[setname][i]
            dialect, reply, first_ms = await one_question(q, history)
            total += 1
            match += dialect == expected
            rows.append((setname, q, expected, dialect, reply, first_ms))
            if setname == "switch":
                history.append({"role": "user", "content": q})
                history.append({"role": "assistant", "content": reply})

    out = ["# DeepSeek dialect samples\n", f"Tag match: {match}/{total}\n"]
    for setname in QUESTIONS:
        out.append(f"\n## {setname}\n")
        out.append("| Question | Expected | Tag | First text (ms) | Reply |")
        out.append("|---|---|---|---|---|")
        for row in rows:
            if row[0] != setname:
                continue
            _, q, expected, dialect, reply, first_ms = row
            first = f"{first_ms:.0f}" if first_ms is not None else "-"
            out.append(f"| {q} | {expected} | {dialect} | {first} | {reply} |")
    (RESULTS / "llm_dialects.md").write_text("\n".join(out) + "\n")
    print(f"wrote {RESULTS / 'llm_dialects.md'} — tag match {match}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
