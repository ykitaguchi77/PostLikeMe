"""TOML-based configuration storage.

Reads and writes the application config at ``<data_dir>/config.toml``.
Supports both full load/save and individual key get/set.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from postlikeme.models.config import AppConfig


class ConfigStore:
    """Manages the PostLikeMe TOML configuration file.

    Parameters
    ----------
    data_dir:
        Root PostLikeMe data directory (typically ``~/.postlikeme``).
        A ``config.toml`` file will be read from / written to this location.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._config_path = data_dir / "config.toml"

    # ------------------------------------------------------------------
    # Full config operations
    # ------------------------------------------------------------------

    def load(self) -> AppConfig:
        """Load the configuration from ``config.toml``.

        If the file does not exist or is empty, a default ``AppConfig``
        is returned (with environment variables resolved).

        Returns
        -------
        AppConfig
            The fully-resolved application configuration.
        """
        if not self._config_path.exists():
            return AppConfig(data_dir=self._data_dir)

        raw_text = self._config_path.read_text(encoding="utf-8")
        if not raw_text.strip():
            return AppConfig(data_dir=self._data_dir)

        data = tomllib.loads(raw_text)

        # The TOML file stores the data_dir as a string; convert to Path.
        if "data_dir" in data and isinstance(data["data_dir"], str):
            data["data_dir"] = Path(data["data_dir"]).expanduser()
        else:
            data["data_dir"] = self._data_dir

        return AppConfig(**data)

    def save(self, config: AppConfig) -> None:
        """Write *config* to ``config.toml``.

        API keys obtained purely from environment variables are **not**
        written to the file to avoid accidentally persisting secrets.
        Only keys that differ from environment-variable values are stored.

        Parameters
        ----------
        config:
            The configuration to persist.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            "# PostLikeMe configuration",
            "# Generated automatically -- feel free to edit manually.",
            "",
        ]

        # -- API keys (only if they do not match the env var) --
        import os

        if config.anthropic_api_key and config.anthropic_api_key != os.environ.get(
            "ANTHROPIC_API_KEY"
        ):
            lines.append(f'anthropic_api_key = "{config.anthropic_api_key}"')
        if config.openai_api_key and config.openai_api_key != os.environ.get("OPENAI_API_KEY"):
            lines.append(f'openai_api_key = "{config.openai_api_key}"')
        if config.x_bearer_token and config.x_bearer_token != os.environ.get("X_BEARER_TOKEN"):
            lines.append(f'x_bearer_token = "{config.x_bearer_token}"')

        # -- LLM settings --
        lines.append(f'default_llm_provider = "{config.default_llm_provider}"')
        lines.append(f'default_llm_model = "{config.default_llm_model}"')

        # -- Collection --
        lines.append(f'default_collector = "{config.default_collector}"')

        # -- Paths --
        lines.append(f'data_dir = "{config.data_dir}"')

        # -- Numeric --
        lines.append(f"default_tweet_count = {config.default_tweet_count}")
        lines.append(f"default_temperature = {config.default_temperature}")

        lines.append("")  # trailing newline

        toml_text = "\n".join(lines)
        self._config_path.write_text(toml_text, encoding="utf-8")

        # Restrict file permissions (may contain API keys).
        try:
            self._config_path.chmod(0o600)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Key-level access
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any:
        """Read a single configuration value by its key name.

        Parameters
        ----------
        key:
            The configuration key, e.g. ``"default_llm_provider"``.

        Returns
        -------
        Any
            The value of the requested key.

        Raises
        ------
        KeyError
            If *key* is not a recognised configuration field.
        """
        config = self.load()
        if not hasattr(config, key):
            known_keys = sorted(config.model_fields.keys())
            raise KeyError(
                f"Unknown config key: {key!r}. Known keys: {', '.join(known_keys)}"
            )
        return getattr(config, key)

    def set(self, key: str, value: Any) -> None:
        """Update a single configuration value and persist to disk.

        Parameters
        ----------
        key:
            The configuration key to update.
        value:
            The new value.  Strings that look like integers or floats will
            be coerced automatically by Pydantic during validation.

        Raises
        ------
        KeyError
            If *key* is not a recognised configuration field.
        ValueError
            If *value* fails Pydantic validation for the field.
        """
        config = self.load()
        if not hasattr(config, key):
            known_keys = sorted(config.model_fields.keys())
            raise KeyError(
                f"Unknown config key: {key!r}. Known keys: {', '.join(known_keys)}"
            )

        # Coerce common string representations
        field_info = config.model_fields[key]
        value = self._coerce_value(value, field_info)

        # Use model_copy to produce a new validated instance
        updated = config.model_copy(update={key: value})
        self.save(updated)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_value(value: Any, field_info: Any) -> Any:
        """Best-effort coercion of CLI string input to the expected type."""
        if not isinstance(value, str):
            return value

        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # Path
        annotation = field_info.annotation
        if annotation is Path or (hasattr(annotation, "__origin__") and annotation is Path):
            return Path(value).expanduser()

        return value
