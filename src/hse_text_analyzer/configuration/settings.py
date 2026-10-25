import os
import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


def _read_secrets_toml() -> dict:
    """Read .streamlit/secrets.toml using stdlib tomllib (Python 3.11+)."""
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        try:
            with open(secrets_path, "rb") as f:
                return tomllib.load(f)
        except Exception:
            pass
    return {}


def _get_secret(key: str, default: str | None = None) -> str | None:
    """Read a secret from env → Streamlit secrets → secrets.toml, in that order."""
    value = os.getenv(key)
    if value:
        return value

    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            value = st.secrets.get(key)
            if value:
                return value
    except Exception:
        pass

    return _read_secrets_toml().get(key, default)


def load_openrouter_secrets() -> tuple[str | None, str, str | None]:
    key = _get_secret("OPENROUTER_API_KEY")
    model = _get_secret("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    extra_model = _get_secret("OPENROUTER_MODEL_EXTRA_1")
    return key, model or "google/gemini-2.5-flash", extra_model


def load_groq_secrets() -> tuple[str | None, str]:
    key = _get_secret("GROQ_API_KEY")
    model = _get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
    return key, model or "openai/gpt-oss-120b"


class AppSettings(BaseModel):
    app_name: str = "HSE Text Analyzer"
    environment: str = Field(default="local")
    log_level: str = Field(default="INFO")
    max_upload_size_mb: int = Field(default=20, gt=0)
    default_language: str = "ru"
    reports_dir: Path = Path("reports")

    # Active provider: readable from ACTIVE_AI_PROVIDER env var or secrets.toml
    active_ai_provider: str = Field(
        default_factory=lambda: _get_secret("ACTIVE_AI_PROVIDER", "openrouter") or "openrouter",
        description="Which AI provider to use: 'openrouter' or 'groq'",
    )

    openrouter_api_key: str | None = Field(default_factory=lambda: load_openrouter_secrets()[0])
    openrouter_model: str = Field(default_factory=lambda: load_openrouter_secrets()[1])
    openrouter_model_extra_1: str | None = Field(default_factory=lambda: load_openrouter_secrets()[2])
    openrouter_api_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions")

    groq_api_key: str | None = Field(default_factory=lambda: load_groq_secrets()[0])
    groq_model: str = Field(default_factory=lambda: load_groq_secrets()[1])
    groq_api_url: str = Field(default="https://api.groq.com/openai/v1/chat/completions")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
