"""Regression gate for CI: fail the build if a key metric drops below tolerance.

Compares eval/results.json against eval/baseline.json. Exit code 1 on regression
so a pull request cannot merge a change that quietly makes quality worse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# metric -> allowed change (negative = how much it may drop before failing)
DEFAULT_THRESHOLDS = {
    "faithfulness": -0.02,
    "answer_relevance": -0.03,
    "recall_at_5": -0.03,
    "refusal_accuracy": -0.05,
}


def compare(current: dict, baseline: dict, thresholds: dict | None = None):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    failures = []
    for metric, tol in thresholds.items():
        cur, base = current.get(metric), baseline.get(metric)
        if cur is None or base is None:
            continue
        if cur - base < tol:
            failures.append((metric, base, cur))
    return len(failures) == 0, failures


def main() -> None:
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    baseline = json.loads((HERE / "baseline.json").read_text(encoding="utf-8"))
    passed, failures = compare(results, baseline)
    if passed:
        print("Regression gate PASSED")
        sys.exit(0)
    print("Regression gate FAILED:")
    for metric, base, cur in failures:
        print(f"  {metric}: baseline {base} -> current {cur}")
    sys.exit(1)


if __name__ == "__main__":
    main()
