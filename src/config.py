from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FAQ_PATH = ROOT_DIR / "data" / "gigacorp_faq.txt"

_PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class MissingAPIKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faq_path: Path = DEFAULT_FAQ_PATH
    retriever_k: int = 3

    @property
    def api_key_env_var(self) -> str:
        return _PROVIDER_KEY_ENV[self.provider]

    @classmethod
    def from_env(cls, provider: str | None = None, api_key: str | None = None) -> "Settings":
        provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        if provider not in _PROVIDER_KEY_ENV:
            raise ValueError(f"Unsupported provider '{provider}'. Choose one of {list(_PROVIDER_KEY_ENV)}.")

        resolved_key = api_key or os.getenv(_PROVIDER_KEY_ENV[provider], "")
        if not resolved_key:
            raise MissingAPIKeyError(
                f"No API key found for provider '{provider}'. "
                f"Set {_PROVIDER_KEY_ENV[provider]} or pass api_key explicitly."
            )
        return cls(provider=provider, api_key=resolved_key)