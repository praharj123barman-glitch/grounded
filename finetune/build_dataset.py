"""Build an instruction dataset from the golden set for optional LoRA fine-tuning.

Each answerable golden question becomes an instruction/response pair. Runs
anywhere, no GPU or API key needed.

  python -m finetune.build_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden_set.jsonl"
OUT = Path(__file__).resolve().parent / "data.jsonl"


def build() -> list[dict]:
    rows: list[dict] = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        if not ex.get("answerable", True):
            continue
        rows.append({"instruction": ex["question"], "response": ex["reference_answer"]})
    OUT.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return rows


if __name__ == "__main__":
    rows = build()
    print(f"Wrote {len(rows)} instruction pairs to {OUT}")
