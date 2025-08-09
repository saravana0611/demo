from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from .config import AssistantSettings
from .prompts import DEFAULT_SYSTEM_PROMPT


def _maybe_bnb_config(use_4bit: bool):
    if not use_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig  # type: ignore

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except Exception:
        return None


def load_model_and_tokenizer(settings: AssistantSettings):
    quant_config = _maybe_bnb_config(settings.use_4bit)

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name_or_path,
        device_map=settings.device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
    )
    model.tie_weights()
    model.eval()
    return model, tokenizer


def build_chat_messages(
    user_prompt: str,
    memory_messages: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if memory_messages:
        messages.extend(memory_messages)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def stream_generate(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> Iterable[str]:
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
        # Fallback: naive concat
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

    thread.join()