"""Optional LoRA fine-tuning. Colab GPU recommended; will not run on CPU.

Fine-tunes a small instruction model on the Grounded Q/A pairs to demonstrate the
workflow. For a document assistant, RAG usually beats fine-tuning (see README);
this exists to show the technique and the RAG-vs-finetune decision, which
interviews ask about.

Colab:
  pip install -q transformers peft trl datasets accelerate bitsandbytes
  python -m finetune.build_dataset
  python finetune/lora_finetune.py
"""
from __future__ import annotations

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # small enough for a free Colab GPU


def main() -> None:
    import json
    from pathlib import Path

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    data_path = Path(__file__).resolve().parent / "data.jsonl"
    rows = [json.loads(x) for x in data_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    dataset = Dataset.from_list(
        [{"text": f"<|user|>\n{r['instruction']}\n<|assistant|>\n{r['response']}"} for r in rows]
    )

    AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir="finetune/out", num_train_epochs=3,
        per_device_train_batch_size=2, learning_rate=2e-4, logging_steps=5,
    )
    trainer = SFTTrainer(model=model, train_dataset=dataset, peft_config=lora, args=args)
    trainer.train()
    trainer.save_model("finetune/out")
    print("Saved LoRA adapter to finetune/out")


if __name__ == "__main__":
    main()
