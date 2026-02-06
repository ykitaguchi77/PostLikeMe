"""CLI command group for generating tweets and replies.

Usage::

    postlikeme generate tweet <username> [--topic TOPIC] [--count N] [--temperature T]
    postlikeme generate reply <username> --to-tweet TEXT [--count N] [--temperature T]
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

generate_app = typer.Typer(
    name="generate",
    help="Generate style-matched tweets and replies.",
)


@generate_app.command("tweet")
def generate_tweet(
    username: str = typer.Argument(
        ...,
        help="Username whose style to emulate (must have a built profile).",
    ),
    topic: str | None = typer.Option(
        None,
        "--topic",
        "-t",
        help="Topic to tweet about. If omitted, a natural topic is chosen.",
    ),
    count: int = typer.Option(
        5,
        "--count",
        "-n",
        help="Number of tweet candidates to generate.",
        min=1,
        max=20,
    ),
    temperature: float = typer.Option(
        0.85,
        "--temperature",
        "--temp",
        help="LLM sampling temperature (0.0-2.0).",
        min=0.0,
        max=2.0,
    ),
    copy: bool = typer.Option(
        False,
        "--copy",
        "-C",
        help="Copy the top-ranked tweet to clipboard.",
    ),
) -> None:
    """Generate tweets in the style of a profiled user."""
    username = username.lstrip("@").lower()

    profile = _load_profile(username)
    if profile is None:
        return

    console.print(
        f"\n[bold]Generating {count} tweet(s) as @{username}[/bold]"
        + (f" about \"{topic}\"" if topic else "")
        + f" (temperature={temperature})\n"
    )

    try:
        ranked = asyncio.run(
            _run_tweet_generation(profile, topic, count, temperature)
        )
    except Exception as exc:
        console.print(f"[red]Generation failed: {exc}[/red]")
        raise typer.Exit(1)

    if not ranked:
        console.print("[yellow]No valid tweets were generated. Try adjusting the temperature.[/yellow]")
        raise typer.Exit(1)

    _display_ranked_results(ranked, "Generated Tweets")

    if copy and ranked:
        _copy_to_clipboard(ranked[0][0])


@generate_app.command("reply")
def generate_reply(
    username: str = typer.Argument(
        ...,
        help="Username whose style to emulate (must have a built profile).",
    ),
    to_tweet: str = typer.Option(
        ...,
        "--to-tweet",
        "-T",
        help="The tweet text to reply to.",
    ),
    count: int = typer.Option(
        5,
        "--count",
        "-n",
        help="Number of reply candidates to generate.",
        min=1,
        max=20,
    ),
    temperature: float = typer.Option(
        0.85,
        "--temperature",
        "--temp",
        help="LLM sampling temperature (0.0-2.0).",
        min=0.0,
        max=2.0,
    ),
    copy: bool = typer.Option(
        False,
        "--copy",
        "-C",
        help="Copy the top-ranked reply to clipboard.",
    ),
) -> None:
    """Generate replies in the style of a profiled user."""
    username = username.lstrip("@").lower()

    profile = _load_profile(username)
    if profile is None:
        return

    console.print(
        f"\n[bold]Generating {count} reply(ies) as @{username}[/bold]\n"
        f"[dim]Replying to: \"{to_tweet}\"[/dim]\n"
    )

    try:
        ranked = asyncio.run(
            _run_reply_generation(profile, to_tweet, count, temperature)
        )
    except Exception as exc:
        console.print(f"[red]Generation failed: {exc}[/red]")
        raise typer.Exit(1)

    if not ranked:
        console.print("[yellow]No valid replies were generated. Try adjusting the temperature.[/yellow]")
        raise typer.Exit(1)

    _display_ranked_results(ranked, "Generated Replies")

    if copy and ranked:
        _copy_to_clipboard(ranked[0][0])


def _load_profile(username: str):
    """Load and return a voice profile, or print an error and exit."""
    from postlikeme.models.config import AppConfig
    from postlikeme.storage.profile_store import ProfileStore

    config = AppConfig()
    profile_store = ProfileStore(config.data_dir)
    profile = profile_store.load_profile(username)

    if profile is None:
        console.print(
            f"[red]No profile found for @{username}. "
            f"Run 'postlikeme collect {username}' and "
            f"'postlikeme analyze {username}' first.[/red]"
        )
        raise typer.Exit(1)

    return profile


async def _run_tweet_generation(
    profile,
    topic: str | None,
    count: int,
    temperature: float,
) -> list[tuple[str, float]]:
    """Run tweet generation and return ranked results."""
    from postlikeme.generation.llm_client import create_llm_client
    from postlikeme.generation.post_processor import PostProcessor
    from postlikeme.generation.prompt_builder import PromptBuilder
    from postlikeme.generation.tweet_generator import TweetGenerator
    from postlikeme.models.config import AppConfig
    from postlikeme.storage.config_store import ConfigStore

    config_store = ConfigStore(AppConfig().data_dir)
    config = config_store.load()
    llm_client = create_llm_client(config)
    prompt_builder = PromptBuilder()
    post_processor = PostProcessor()
    generator = TweetGenerator(llm_client, prompt_builder, post_processor)

    return await generator.generate_ranked(profile, topic, count, temperature)


async def _run_reply_generation(
    profile,
    target_tweet: str,
    count: int,
    temperature: float,
) -> list[tuple[str, float]]:
    """Run reply generation and return ranked results."""
    from postlikeme.generation.llm_client import create_llm_client
    from postlikeme.generation.post_processor import PostProcessor
    from postlikeme.generation.prompt_builder import PromptBuilder
    from postlikeme.generation.reply_generator import ReplyGenerator
    from postlikeme.models.config import AppConfig
    from postlikeme.storage.config_store import ConfigStore

    config_store = ConfigStore(AppConfig().data_dir)
    config = config_store.load()
    llm_client = create_llm_client(config)
    prompt_builder = PromptBuilder()
    post_processor = PostProcessor()
    generator = ReplyGenerator(llm_client, prompt_builder, post_processor)

    return await generator.generate_ranked(profile, target_tweet, count, temperature)


def _display_ranked_results(ranked: list[tuple[str, float]], title: str) -> None:
    """Display ranked tweet/reply candidates in a Rich table."""
    table = Table(title=title, show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Text", ratio=4)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Length", justify="right", width=6)

    for i, (text, score) in enumerate(ranked, 1):
        # Color the score based on quality
        if score >= 0.8:
            score_str = f"[green]{score:.2f}[/green]"
        elif score >= 0.6:
            score_str = f"[yellow]{score:.2f}[/yellow]"
        else:
            score_str = f"[red]{score:.2f}[/red]"

        # Highlight the top pick
        rank_str = f"[bold green]{i}[/bold green]" if i == 1 else str(i)

        table.add_row(rank_str, text, score_str, str(len(text)))

    console.print(table)
    console.print()

    # Also display the top result in a highlighted panel
    if ranked:
        best_text, best_score = ranked[0]
        console.print(
            Panel(
                best_text,
                title=f"[bold green]Top Pick (score: {best_score:.2f})[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )


def _copy_to_clipboard(text: str) -> None:
    """Attempt to copy text to the system clipboard."""
    try:
        import subprocess

        process = subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode(),
            capture_output=True,
            timeout=5,
        )
        if process.returncode == 0:
            console.print("[dim]Copied to clipboard.[/dim]")
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        import subprocess

        process = subprocess.run(
            ["xsel", "--clipboard", "--input"],
            input=text.encode(),
            capture_output=True,
            timeout=5,
        )
        if process.returncode == 0:
            console.print("[dim]Copied to clipboard.[/dim]")
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        import subprocess

        process = subprocess.run(
            ["pbcopy"],
            input=text.encode(),
            capture_output=True,
            timeout=5,
        )
        if process.returncode == 0:
            console.print("[dim]Copied to clipboard.[/dim]")
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    console.print("[dim]Could not copy to clipboard (no clipboard tool found).[/dim]")
