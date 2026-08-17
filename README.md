# Grounded

An agentic RAG research assistant that answers questions over a document corpus,
cites the exact source for every claim, and refuses when the context does not
support an answer. The headline is not the chatbot; it is a reproducible
**evaluation harness** that scores every change (faithfulness, answer relevance,
retrieval recall@k, latency, cost) and gates regressions in CI.

Corpus: Indian company annual reports (swappable via config).

## Status

Live-verified end to end. **31 offline tests pass**; ingestion, retrieval,
grounded answering, the eval harness, the multi-agent flow, and the MCP server all
run. Real numbers on the sample corpus: **recall@5 = 1.0, MRR = 0.94,
faithfulness = 1.0** (see `eval/baseline.json`).

Covers the full AI Engineer course syllabus, all 8 weeks:

- [x] Week 1 — LLM fundamentals + API mastery (multi-provider client, cost logging)
- [x] Week 2 — prompt engineering + structured outputs (Pydantic answer, defensive prompts)
- [x] Week 3 — RAG foundations (ingest, chunking, embeddings, Chroma)
- [x] Week 4 — advanced RAG + evaluation (hybrid, rerank, query rewrite, LLM-judge, regression gate)
- [x] Week 5 — agents + tool use (LangGraph agent, calculator tool)
- [x] Week 6 — LangGraph + MCP + multi-agent (planner/critic graph, MCP server)
- [x] Week 7 — observability + guardrails + security (Langfuse shim, injection/PII guard, semantic cache)
- [x] Week 8 — deployment + fine-tuning (FastAPI, Docker, CI; LoRA script + RAG-vs-finetune framework)

Remaining polish: fill the full 14-question generation eval on fresh quota, run the
ablation table, deploy to a live URL.

## Quickstart

```powershell
# 1. environment (Windows PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. run the offline test suite (no API key needed)
pytest -q

# 3. verify the pipeline offline on the shipped sample report
python -m grounded.ingest --dry-run --source sample

# 4. add your key, build the index, then ask a question
copy .env.example .env          # paste your Gemini key into .env
python -m grounded.ingest --source sample
python -m grounded.ask "What was Meridian's FY2025 revenue?"

# 5. run the evaluation suite and the regression gate (run as modules)
python -m eval.run_evals                  # add --retrieval-only to skip chat (quota-safe)
python -m eval.regression_gate

# 6. serve the API
uvicorn grounded.api.app:app --reload

# 7. multi-agent (planner, retrieve, answer, critic), MCP server, fine-tune dataset
python -m grounded.ask_agent "Compare Meridian's revenue and profit growth."
python -m grounded.mcp_server.server        # exposes retrieval + answering as MCP tools
python -m finetune.build_dataset            # LoRA training itself runs on Colab
```

Get a free Gemini key (no card) at https://aistudio.google.com/apikey

## Layout

```
grounded/
  grounded/
    config.py            # every tunable (models, chunking, retrieval flags)
    llm/                 # provider-agnostic client, cost pricing, answer schema
    ingestion/           # loader, chunker, embeddings, Chroma indexer
    retrieval/           # dense, hybrid (BM25 + RRF), reranking, query rewrite
    rag/                 # prompts, answerer, pipeline
    agent/               # LangGraph agent, tools, and multi-agent (planner/critic)
    guardrails/          # prompt-injection and PII input guard
    cache/               # semantic cache
    observability/       # Langfuse tracer shim (no-op if unconfigured)
    api/                 # FastAPI app (/health, /ask)
    mcp_server/          # MCP server exposing retrieval + answering as tools
    ingest.py / ask.py / ask_agent.py   # CLIs
  eval/
    golden_set.jsonl     # curated Q/A over the sample report (incl. unanswerables)
    metrics.py           # recall@k, precision@k, MRR, hit@k
    judge.py             # LLM-as-judge faithfulness + relevance
    run_evals.py         # runs the suite, writes results.json
    regression_gate.py   # fails CI if a metric regresses vs baseline.json
  finetune/              # Week 8: build_dataset + LoRA script (Colab) + RAG-vs-finetune doc
  tests/                 # 31 offline tests (deterministic fakes, no key needed)
  data/{raw,sample}/     # drop PDFs into raw/; sample/ ships an excerpt
  Dockerfile
  .github/workflows/evals.yml
```

## Stack

Python, LangChain, LangGraph, Google Gemini, Chroma (Qdrant next), RAGAS-style
LLM-as-judge, Langfuse, FastAPI, Docker, GitHub Actions. Aligned to the
"Padho with Pratyush" AI Engineer curriculum, with an evaluation-first layer on top.
