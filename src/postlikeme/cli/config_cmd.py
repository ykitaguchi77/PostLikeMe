"""CLI commands for managing PostLikeMe configuration.

Usage::

    postlikeme config init
    postlikeme config show
    postlikeme config set <key> <value>
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

config_app = typer.Typer(
    name="config",
    help="Manage PostLikeMe configuration.",
)


def _get_config_store():
    """Create and return a ConfigStore using the default data directory."""
    from postlikeme.models.config import AppConfig
    from postlikeme.storage.config_store import ConfigStore

    default_config = AppConfig()
    default_config.ensure_dirs()
    return ConfigStore(default_config.data_dir)


def _mask_key(value: str | None) -> str:
    """Mask an API key, showing only the first 4 characters.

    Examples:
        "sk-abc123xyz789" -> "sk-a***"
        None -> "(not set)"
        "" -> "(not set)"
    """
    if not value:
        return "(not set)"
    if len(value) <= 4:
        return value[0] + "***"
    return value[:4] + "***"


@config_app.command("init")
def init() -> None:
    """Interactively configure PostLikeMe API keys and settings."""
    console.print()
    console.print(
        Panel(
            "Welcome to PostLikeMe configuration.\n"
            "You will be prompted for API keys and settings.\n"
            "Press Enter to skip any field and keep the current/default value.",
            title="PostLikeMe Setup",
            border_style="blue",
        )
    )
    console.print()

    config_store = _get_config_store()
    config = config_store.load()

    # --- Anthropic API Key ---
    current_anthropic = _mask_key(config.anthropic_api_key)
    console.print(f"[dim]Current Anthropic API key: {current_anthropic}[/dim]")
    anthropic_key = typer.prompt(
        "Anthropic API key (for Claude models)",
        default="",
        show_default=False,
    ).strip()
    if anthropic_key:
        config = config.model_copy(update={"anthropic_api_key": anthropic_key})

    # --- OpenAI API Key ---
    current_openai = _mask_key(config.openai_api_key)
    console.print(f"\n[dim]Current OpenAI API key: {current_openai}[/dim]")
    openai_key = typer.prompt(
        "OpenAI API key (for GPT models)",
        default="",
        show_default=False,
    ).strip()
    if openai_key:
        config = config.model_copy(update={"openai_api_key": openai_key})

    # --- X/Twitter Bearer Token ---
    current_x = _mask_key(config.x_bearer_token)
    console.print(f"\n[dim]Current X/Twitter Bearer Token: {current_x}[/dim]")
    x_token = typer.prompt(
        "X/Twitter API Bearer Token",
        default="",
        show_default=False,
    ).strip()
    if x_token:
        config = config.model_copy(update={"x_bearer_token": x_token})

    # --- LLM Provider ---
    console.print(f"\n[dim]Current LLM provider: {config.default_llm_provider}[/dim]")
    provider = typer.prompt(
        "Default LLM provider (anthropic/openai)",
        default=config.default_llm_provider,
    ).strip().lower()
    if provider in ("anthropic", "openai"):
        config = config.model_copy(update={"default_llm_provider": provider})
    elif provider:
        console.print(
            f"[yellow]Unknown provider '{provider}'; keeping '{config.default_llm_provider}'.[/yellow]"
        )

    # --- LLM Model ---
    console.print(f"\n[dim]Current LLM model: {config.default_llm_model}[/dim]")
    model = typer.prompt(
        "Default LLM model",
        default=config.default_llm_model,
    ).strip()
    if model:
        config = config.model_copy(update={"default_llm_model": model})

    # --- Default tweet count ---
    console.print(f"\n[dim]Current default tweet count: {config.default_tweet_count}[/dim]")
    count_str = typer.prompt(
        "Default tweet count for collection",
        default=str(config.default_tweet_count),
    ).strip()
    try:
        count_val = int(count_str)
        if count_val >= 1:
            config = config.model_copy(update={"default_tweet_count": count_val})
    except ValueError:
        console.print("[yellow]Invalid number; keeping current value.[/yellow]")

    # --- Default temperature ---
    console.print(f"\n[dim]Current default temperature: {config.default_temperature}[/dim]")
    temp_str = typer.prompt(
        "Default generation temperature (0.0-2.0)",
        default=str(config.default_temperature),
    ).strip()
    try:
        temp_val = float(temp_str)
        if 0.0 <= temp_val <= 2.0:
            config = config.model_copy(update={"default_temperature": temp_val})
        else:
            console.print("[yellow]Temperature out of range; keeping current value.[/yellow]")
    except ValueError:
        console.print("[yellow]Invalid number; keeping current value.[/yellow]")

    # Save configuration
    config_store.save(config)

    console.print()
    console.print(
        f"[green]Configuration saved to: {config_store._config_path}[/green]"
    )
    console.print()


@config_app.command("show")
def show() -> None:
    """Display the current configuration with masked API keys."""
    config_store = _get_config_store()
    config = config_store.load()

    table = Table(title="PostLikeMe Configuration", show_header=True)
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    # API Keys (masked)
    table.add_row("anthropic_api_key", _mask_key(config.anthropic_api_key))
    table.add_row("openai_api_key", _mask_key(config.openai_api_key))
    table.add_row("x_bearer_token", _mask_key(config.x_bearer_token))

    # LLM settings
    table.add_row("default_llm_provider", config.default_llm_provider)
    table.add_row("default_llm_model", config.default_llm_model)

    # Collection settings
    table.add_row("default_collector", config.default_collector)
    table.add_row("default_tweet_count", str(config.default_tweet_count))

    # Generation settings
    table.add_row("default_temperature", str(config.default_temperature))

    # Paths
    table.add_row("data_dir", str(config.data_dir))
    table.add_row("profiles_dir", str(config.profiles_dir))
    table.add_row("cache_dir", str(config.cache_dir))
    table.add_row("config_path", str(config.config_path))

    console.print()
    console.print(table)
    console.print()


@config_app.command("set")
def set_value(
    key: str = typer.Argument(
        ...,
        help="Configuration key to set (e.g. 'default_llm_provider', 'anthropic_api_key').",
    ),
    value: str = typer.Argument(
        ...,
        help="Value to set for the key.",
    ),
) -> None:
    """Set a single configuration value."""
    config_store = _get_config_store()

    try:
        config_store.set(key, value)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except (ValueError, Exception) as exc:
        console.print(f"[red]Failed to set '{key}': {exc}[/red]")
        raise typer.Exit(1)

    # Display the updated value (masked if it's an API key)
    display_value = value
    if "api_key" in key or "token" in key or "bearer" in key:
        display_value = _mask_key(value)

    console.print(f"[green]Set {key} = {display_value}[/green]")
