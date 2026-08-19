# Grounded

**Live demo:** https://grounded-7vi4.onrender.com/docs (interactive API) ·
try `https://grounded-7vi4.onrender.com/ask?q=What+was+Meridian's+FY2025+revenue`

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

What is implemented:

- Provider-agnostic LLM client with token, cost, and latency logging
- Structured, typed answers (Pydantic) with per-claim citations and a calibrated refusal path
- Configurable ingestion: loader, recursive chunking, embeddings, Chroma index
- Hybrid retrieval: dense embeddings + BM25 keyword, fused with Reciprocal Rank Fusion
- Optional LLM reranking and multi-query rewriting, all ablatable from config
- Evaluation harness: recall@k, precision@k, MRR, hit@k, LLM-as-judge faithfulness and answer relevance, and a CI regression gate
- Agentic layer: a LangGraph tool-using agent and a multi-agent planner / retriever / answerer / critic graph
- Guardrails (prompt-injection + PII), a semantic cache, and a Langfuse tracer shim
- FastAPI service, Docker image, GitHub Actions CI, and an MCP server
- Optional LoRA fine-tuning script with a RAG-vs-fine-tuning decision writeup

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
  finetune/              # build_dataset + LoRA script (Colab) + RAG-vs-finetune doc
  tests/                 # 31 offline tests (deterministic fakes, no key needed)
  data/{raw,sample}/     # drop PDFs into raw/; sample/ ships an excerpt
  Dockerfile
  .github/workflows/evals.yml
```

## Stack

Python, LangChain, LangGraph, Google Gemini, Chroma (Qdrant next), RAGAS-style
LLM-as-judge, Langfuse, FastAPI, Docker, GitHub Actions. Retrieval quality is
measured and gated on every change, not assumed.
