"""Run the full evaluation suite over the golden set and write results.json.

  python eval/run_evals.py                 # uses config's retrieval settings
  python eval/run_evals.py --limit 5       # quick subset

Requires GOOGLE_API_KEY and a built index (run `python -m grounded.ingest` first).
Retrieval metrics always run; generation metrics (LLM judge) run on answerable
examples. Operational metrics (latency) are recorded per query.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from grounded.config import settings
from grounded.rag.prompts import build_context
from eval import metrics as M

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden_set.jsonl"
RESULTS = HERE / "results.json"
K = 5


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retrieval-only", action="store_true",
                    help="compute only retrieval metrics (embeddings), no chat calls (quota-safe)")
    args = ap.parse_args()

    if not settings.google_api_key:
        print("GOOGLE_API_KEY is empty. Add it to .env, run ingestion, then re-run.")
        return

    from grounded.llm.client import LLMClient
    from grounded.rag.pipeline import build_default_pipeline
    from eval.judge import Judge

    pipe = build_default_pipeline()
    judge = None if args.retrieval_only else Judge(LLMClient())

    golden = load_golden()
    if args.limit:
        golden = golden[: args.limit]

    recalls, precisions, mrrs, hits = [], [], [], []
    faiths, rels, latencies = [], [], []
    refusal_correct: list[float] = []

    for ex in golden:
        t0 = time.time()
        if args.retrieval_only:
            answer, chunks = None, pipe.retrieve(ex["question"], k=K)
        else:
            answer, chunks = pipe.run(ex["question"], k=K)
        latencies.append(time.time() - t0)
        texts = [c.text for c in chunks]
        subs = ex.get("relevant_substrings", [])

        recalls.append(M.recall_at_k(texts, subs, K))
        precisions.append(M.precision_at_k(texts, subs, K))
        mrrs.append(M.mrr(texts, subs))
        hits.append(M.hit_at_k(texts, subs, K))

        if not args.retrieval_only:
            if ex.get("answerable", True):
                scores = judge.score(ex["question"], answer.answer, build_context(chunks))
                faiths.append(scores.faithfulness)
                rels.append(scores.answer_relevance)
            else:
                # correct behaviour on an unanswerable question is to refuse
                refusal_correct.append(1.0 if not answer.answered else 0.0)

    aggregate = {
        "recall_at_5": round(M.mean(recalls), 4),
        "precision_at_5": round(M.mean(precisions), 4),
        "mrr": round(M.mean(mrrs), 4),
        "hit_at_5": round(M.mean(hits), 4),
        "faithfulness": round(M.mean(faiths), 4),
        "answer_relevance": round(M.mean(rels), 4),
        "refusal_accuracy": round(M.mean(refusal_correct), 4),
        "p50_latency_s": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else 0.0,
        "examples": len(golden),
        "config": {
            "use_hybrid": settings.use_hybrid,
            "use_query_rewrite": settings.use_query_rewrite,
            "use_rerank": settings.use_rerank,
            "chunk_size": settings.chunk_size,
        },
    }
    RESULTS.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
