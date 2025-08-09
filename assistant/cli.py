from __future__ import annotations

import sys

from .config import AssistantSettings
from .memory import ConversationMemory
from .model_loader import build_chat_messages, load_model_and_tokenizer, stream_generate
from .prompts import DEFAULT_SYSTEM_PROMPT


def main() -> None:
    settings = AssistantSettings.from_env_and_yaml()
    model, tokenizer = load_model_and_tokenizer(settings)
    memory = ConversationMemory(max_turns=settings.memory_turns)

    system_prompt = settings.system_prompt or DEFAULT_SYSTEM_PROMPT

    print("Company Assistant CLI. Type 'exit' to quit.\n", flush=True)
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        messages = build_chat_messages(user, memory_messages=memory.get(), system_prompt=system_prompt)
        for piece in stream_generate(
            model,
            tokenizer,
            messages,
            max_new_tokens=settings.generation.max_new_tokens,
            temperature=settings.generation.temperature,
            top_p=settings.generation.top_p,
            repetition_penalty=settings.generation.repetition_penalty,
            do_sample=settings.generation.do_sample,
        ):
            sys.stdout.write(piece)
            sys.stdout.flush()
        print()
        memory.add("user", user)
        # The last assistant message is everything generated; for simplicity, we do not re-tokenize.
        # In production, use the server to maintain exact assistant outputs.
        # Here, we keep the rolling memory conceptually.
        memory.add("assistant", "")


if __name__ == "__main__":
    main()