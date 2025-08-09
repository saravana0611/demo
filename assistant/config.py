from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationConfig(BaseModel):
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    do_sample: bool = True


class AssistantSettings(BaseSettings):
    model_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_4bit: bool = False
    device_map: str = "auto"

    # memory
    memory_turns: int = 16

    # prompts
    system_prompt: Optional[str] = None

    # generation
    generation: GenerationConfig = GenerationConfig()

    # optional yaml config file
    config_yaml: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="ASSISTANT_",
        env_nested_delimiter="_",
        extra="ignore",
    )

    @classmethod
    def from_env_and_yaml(cls) -> "AssistantSettings":
        tmp = cls()
        yaml_path = tmp.config_yaml
        if yaml_path and Path(yaml_path).exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            merged = cls._merge_yaml_into_env(tmp, data)
            return merged
        return tmp

    @staticmethod
    def _merge_yaml_into_env(env_settings: "AssistantSettings", data: Dict[str, Any]) -> "AssistantSettings":
        payload: Dict[str, Any] = env_settings.model_dump()
        for key, value in data.items():
            if key == "generation" and isinstance(value, dict):
                payload[key] = {**payload.get(key, {}), **value}
            else:
                payload[key] = value
        return AssistantSettings(**payload)