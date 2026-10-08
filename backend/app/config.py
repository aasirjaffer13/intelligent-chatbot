"""Typed application settings loaded from environment / .env files.

Every secret and deployment knob lives in the environment — never in source.
Variables are read with the ``NOVA_`` prefix (e.g. ``NOVA_PORT=8000``).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the NOVA API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NOVA_",
        extra="ignore",
        populate_by_name=True,
    )

    env: str = Field(default="development")
    app_name: str = Field(default="NOVA")
    debug: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        description="Comma-separated allowed browser origins.",
    )
    log_level: str = Field(default="INFO")
    intent_backend: str = Field(
        default="auto",
        description="Intent classifier backend: auto | sklearn | embedding | keyword.",
    )
    database_url: str = Field(
        default="",
        description=(
            "SQLAlchemy URL for conversation memory. Empty = in-memory store. "
            "Production: postgresql+psycopg://user:pass@host:5432/dbname"
        ),
    )
    memory_window: int = Field(
        default=12,
        ge=0,
        le=100,
        description="How many recent messages form the session context window.",
    )
    rag_top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="How many chunks retrieval returns per document question.",
    )
    rag_min_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity for a chunk to ground an answer; "
            "below it the bot refuses instead of guessing."
        ),
    )
    max_upload_mb: int = Field(default=10, ge=1, le=100, description="Upload size cap.")
    document_dir: str = Field(
        default="",
        description="Where uploaded files are stored. Empty = nova/data/documents.",
    )

    # --- Phase 8: LLM reply generation ---
    llm_provider: str = Field(
        default="auto",
        description=(
            "LLM provider: auto (key-aware) | openai | huggingface | local | "
            "mock | none (templates only)."
        ),
    )
    llm_model: str = Field(
        default="",
        description="Model name override; empty = provider default.",
    )
    llm_max_tokens: int = Field(
        default=256, ge=16, le=4096, description="Max tokens per completion."
    )
    llm_temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="Sampling temperature (0 = deterministic)."
    )
    llm_timeout: float = Field(
        default=30.0, gt=0.0, le=300.0, description="Provider HTTP timeout in seconds."
    )
    # --- Phase 9: agent loop ---
    agent_max_steps: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Max LLM/tool rounds per turn before the loop bails out.",
    )
    # --- Phase 10: streaming ---
    stream_delay_ms: int = Field(
        default=15,
        ge=0,
        le=200,
        description="Delay between streamed reply chunks (0 = stream as fast as possible).",
    )
    # Secrets: standard env names first, NOVA_-prefixed alternates second.
    # Never defaulted in .env.example, never logged, never repr'd.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "NOVA_OPENAI_API_KEY"),
    )
    hf_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "NOVA_HF_TOKEN"
        ),
    )

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("intent_backend")
    @classmethod
    def _lower_intent_backend(cls, value: str) -> str:
        allowed = {"auto", "sklearn", "embedding", "keyword"}
        lowered = value.lower()
        if lowered not in allowed:
            raise ValueError(f"intent_backend must be one of {sorted(allowed)}")
        return lowered

    @field_validator("llm_provider")
    @classmethod
    def _lower_llm_provider(cls, value: str) -> str:
        allowed = {"auto", "openai", "huggingface", "local", "mock", "none"}
        lowered = value.lower()
        if lowered not in allowed:
            raise ValueError(f"llm_provider must be one of {sorted(allowed)}")
        return lowered

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse the comma-separated origin string into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so the app reads config exactly once."""
    return Settings()
