from __future__ import annotations

from typing import Dict, List


class ConversationMemory:
    def __init__(self, max_turns: int = 16) -> None:
        self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        excess = len(self.messages) - (self.max_turns * 2 + 2)
        if excess > 0:
            self.messages = self.messages[excess:]

    def get(self) -> List[Dict[str, str]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()