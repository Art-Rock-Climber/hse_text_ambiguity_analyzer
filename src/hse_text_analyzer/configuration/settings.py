import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


def load_openrouter_secrets() -> tuple[str | None, str, str | None]:
    key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    extra_model = os.getenv("OPENROUTER_MODEL_EXTRA_1")
    
    if not key or not model or not extra_model:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and st.secrets:
                if not key:
                    key = st.secrets.get("OPENROUTER_API_KEY")
                if not model:
                    model = st.secrets.get("OPENROUTER_MODEL")
                if not extra_model:
                    extra_model = st.secrets.get("OPENROUTER_MODEL_EXTRA_1")
        except Exception:
            pass
            
    if not key or not model or not extra_model:
        try:
            import toml
            secrets_path = Path(".streamlit/secrets.toml")
            if secrets_path.exists():
                data = toml.load(secrets_path)
                if not key:
                    key = data.get("OPENROUTER_API_KEY")
                if not model:
                    model = data.get("OPENROUTER_MODEL")
                if not extra_model:
                    extra_model = data.get("OPENROUTER_MODEL_EXTRA_1")
        except Exception:
            pass
            
    return key, model or "google/gemini-2.5-flash", extra_model


def load_groq_secrets() -> tuple[str | None, str]:
    key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")
    
    if not key or not model:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and st.secrets:
                if not key:
                    key = st.secrets.get("GROQ_API_KEY")
                if not model:
                    model = st.secrets.get("GROQ_MODEL")
        except Exception:
            pass
            
    if not key or not model:
        try:
            import toml
            secrets_path = Path(".streamlit/secrets.toml")
            if secrets_path.exists():
                data = toml.load(secrets_path)
                if not key:
                    key = data.get("GROQ_API_KEY")
                if not model:
                    model = data.get("GROQ_MODEL")
        except Exception:
            pass
            
    return key, model or "llama3-70b-8192"


class AppSettings(BaseModel):
    app_name: str = "HSE Text Analyzer"
    environment: str = Field(default="local")
    log_level: str = Field(default="INFO")
    max_upload_size_mb: int = Field(default=20, gt=0)
    default_language: str = "ru"
    reports_dir: Path = Path("reports")
    
    active_ai_provider: str = Field(default="openrouter", description="Which AI provider to use: 'openrouter' or 'groq'")
    
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
