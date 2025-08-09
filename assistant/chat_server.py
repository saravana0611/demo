from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import AssistantSettings
from .memory import ConversationMemory
from .model_loader import build_chat_messages, load_model_and_tokenizer, stream_generate
from .prompts import DEFAULT_SYSTEM_PROMPT

app = FastAPI(title="Company LLM Assistant")

_settings = AssistantSettings.from_env_and_yaml()
_model_tokenizer: Optional[Tuple[object, object]] = None
_memory = ConversationMemory(max_turns=_settings.memory_turns)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = None
    user: Optional[str] = None
    system: Optional[str] = None

    stream: bool = True

    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None


def _iter_stream(text_stream: Iterable[str]):
    for chunk in text_stream:
        yield chunk


def _ensure_model_loaded():
    global _model_tokenizer
    if _model_tokenizer is None:
        _model_tokenizer = load_model_and_tokenizer(_settings)
    return _model_tokenizer


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model_loaded": _model_tokenizer is not None}


@app.post("/chat")
async def chat(req: ChatRequest):
    (model, tokenizer) = _ensure_model_loaded()

    gen_cfg = _settings.generation.model_copy()
    if req.max_new_tokens is not None:
        gen_cfg.max_new_tokens = req.max_new_tokens
    if req.temperature is not None:
        gen_cfg.temperature = req.temperature
    if req.top_p is not None:
        gen_cfg.top_p = req.top_p
    if req.repetition_penalty is not None:
        gen_cfg.repetition_penalty = req.repetition_penalty
    if req.do_sample is not None:
        gen_cfg.do_sample = req.do_sample

    if req.messages is not None:
        messages = [m.model_dump() for m in req.messages]
    else:
        if not req.user:
            raise HTTPException(status_code=400, detail="Either messages or user must be provided")
        system_prompt = req.system or _settings.system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = build_chat_messages(
            req.user,
            memory_messages=_memory.get(),
            system_prompt=system_prompt,
        )

    text_stream = stream_generate(
        model,
        tokenizer,
        messages,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        repetition_penalty=gen_cfg.repetition_penalty,
        do_sample=gen_cfg.do_sample,
    )

    if req.stream:
        return StreamingResponse(_iter_stream(text_stream), media_type="text/plain")
    else:
        final_text = "".join(list(text_stream))
        return JSONResponse({"content": final_text})