"""Application configuration model.

Configuration is persisted as TOML at ``~/.postlikeme/config.toml`` and can be
overridden by environment variables or CLI flags.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _default_data_dir() -> Path:
    """Return the default data directory: ``~/.postlikeme``."""
    return Path.home() / ".postlikeme"


class AppConfig(BaseModel):
    """Top-level application configuration.

    Values are resolved in order of precedence:
    1. Explicit CLI flags (highest)
    2. Environment variables
    3. ``config.toml`` file
    4. Defaults defined here (lowest)
    """

    # --- API keys (optional; can come from env vars) ---
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key. Falls back to OPENAI_API_KEY env var.",
    )
    x_bearer_token: str | None = Field(
        default=None,
        description="X/Twitter API v2 Bearer Token. Falls back to X_BEARER_TOKEN env var.",
    )

    # --- LLM settings ---
    default_llm_provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description="Which LLM backend to use by default.",
    )
    default_llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default model identifier for the chosen LLM provider.",
    )

    # --- Collection settings ---
    default_collector: Literal["api", "scrape", "archive"] = Field(
        default="api",
        description="Default tweet collection backend.",
    )

    # --- Paths ---
    data_dir: Path = Field(
        default_factory=_default_data_dir,
        description="Root directory for all PostLikeMe data "
        "(profiles, cache, logs). Defaults to ~/.postlikeme.",
    )

    # --- Analysis / generation defaults ---
    default_tweet_count: int = Field(
        default=500,
        ge=1,
        description="Default number of tweets to collect for analysis.",
    )
    default_temperature: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        description="Default LLM sampling temperature for generation.",
    )

    # ------------------------------------------------------------------
    # Derived paths (computed, not stored)
    # ------------------------------------------------------------------

    @property
    def profiles_dir(self) -> Path:
        """Directory where voice profile JSON files are stored."""
        return self.data_dir / "profiles"

    @property
    def cache_dir(self) -> Path:
        """Directory for the SQLite tweet cache."""
        return self.data_dir / "cache"

    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        return self.data_dir / "logs"

    @property
    def config_path(self) -> Path:
        """Path to the TOML configuration file."""
        return self.data_dir / "config.toml"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _resolve_env_vars(self) -> AppConfig:
        """Fill in API keys from environment variables when not set explicitly."""
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.x_bearer_token is None:
            self.x_bearer_token = os.environ.get("X_BEARER_TOKEN")
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create all required directories if they do not exist."""
        for directory in (self.data_dir, self.profiles_dir, self.cache_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        # Restrict permissions on the data directory (contains API keys in config.toml).
        try:
            self.data_dir.chmod(0o700)
        except OSError:
            pass  # Best-effort; may fail on some filesystems.
