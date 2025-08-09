from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import datasets as hfds
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


@dataclass
class Record:
    system: Optional[str]
    user: str
    assistant: str


def format_example(rec: Dict[str, str], tokenizer) -> str:
    messages: List[Dict[str, str]] = []
    if rec.get("system"):
        messages.append({"role": "system", "content": rec["system"]})
    messages.append({"role": "user", "content": rec["user"]})
    messages.append({"role": "assistant", "content": rec["assistant"]})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return f"system: {rec.get('system','')}\nuser: {rec['user']}\nassistant: {rec['assistant']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dataset_path", required=True, help="JSONL with keys: user, assistant, optional system")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    dataset = hfds.load_dataset("json", data_files=args.dataset_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def map_fn(batch):
        texts = [format_example(rec, tokenizer) for rec in batch]
        return {"text": texts}

    dataset = dataset.map(map_fn, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_cfg,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=max(1, 64 // args.batch_size),
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            max_seq_length=args.max_seq_len,
            logging_steps=10,
            save_steps=1000,
            save_total_limit=3,
            fp16=False,
            bf16=True,
            report_to=[],
        ),
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()