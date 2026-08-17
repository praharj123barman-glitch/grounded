# Fine-tuning (course Week 8)

This folder covers the Week 8 fine-tuning topic. Two parts: the decision (when to
fine-tune at all) and a working LoRA script you run on a free Colab GPU.

## When to fine-tune vs when to use RAG (the interview question)

Reach for **RAG** (what Grounded does) when the goal is to ground answers in
documents that change, and you need citations and freshness. RAG adds knowledge
without retraining.

Reach for **fine-tuning** when you need to change *behaviour or style* rather than
add facts: a consistent output format, a domain tone, a classification head, or to
compress a long prompt into learned behaviour. Fine-tuning does not reliably add
new factual knowledge, and stale facts get baked in.

For a document assistant, RAG almost always wins. This folder is here to show the
workflow and, more importantly, that you can reason about the trade-off, which
interviews ask about directly.

## How to run (Colab GPU, not local CPU)

```bash
pip install -q transformers peft trl datasets accelerate bitsandbytes
python -m finetune.build_dataset      # turns the golden set into instruction pairs
python finetune/lora_finetune.py      # trains a LoRA adapter on a small model
```

`build_dataset.py` runs anywhere (no GPU). `lora_finetune.py` needs a GPU, so run
it on Colab; it trains a LoRA adapter on `Qwen/Qwen2.5-0.5B-Instruct` and saves it
to `finetune/out`.
