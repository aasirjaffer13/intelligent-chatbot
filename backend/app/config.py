"""Typed application settings loaded from environment / .env files.

Every secret and deployment knob lives in the environment — never in source.
Variables are read with the ``NOVA_`` prefix (e.g. ``NOVA_PORT=8000``).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the NOVA API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NOVA_",
        extra="ignore",
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
