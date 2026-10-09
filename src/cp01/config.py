"""Validated runtime-only configuration. API keys are excluded from representations."""

from dataclasses import dataclass, field
import os

from .transport import RetryPolicy


@dataclass(frozen=True)
class APIConfig:
    base_url: str
    model: str = "llama3.2"
    fallback_provider: str = "none"
    fallback_model: str = ""
    fallback_key: str = field(default="", repr=False)
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    @classmethod
    def from_env(cls):
        fallback = os.getenv("FALLBACK_PROVIDER", "none").strip().lower()
        if fallback not in {"none", "openai", "anthropic"}:
            raise ValueError("FALLBACK_PROVIDER must be none, openai, or anthropic.")
        key, model = "", ""
        if fallback != "none":
            prefix = "OPENAI" if fallback == "openai" else "ANTHROPIC"
            key = os.getenv(prefix + "_API_KEY", "").strip()
            model = os.getenv(prefix + "_MODEL", "").strip()
            if not key or not model:
                raise ValueError(
                    f"Opt-in fallback requires {prefix}_API_KEY and {prefix}_MODEL in your private .env."
                )
        return cls(
            os.getenv("HOSTED_LLM_BASE_URL", "").strip(),
            os.getenv("HOSTED_LLM_MODEL", "llama3.2"),
            fallback,
            model,
            key,
        )
