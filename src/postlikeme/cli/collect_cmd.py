"""CLI command for collecting tweets from an X/Twitter account.

Usage::

    postlikeme collect <username> [--count 500] [--collector api] [--include-replies]
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def collect(
    username: str = typer.Argument(
        ...,
        help="X/Twitter username to collect tweets from (without @ prefix).",
    ),
    count: int = typer.Option(
        500,
        "--count",
        "-n",
        help="Maximum number of tweets to collect.",
        min=1,
        max=10000,
    ),
    collector: str = typer.Option(
        "api",
        "--collector",
        "-c",
        help="Collection backend: 'api' (Tweepy), 'scrape' (twscrape), or 'archive' (JSON file).",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a Twitter archive JSON file (required when --collector=archive).",
        exists=False,
    ),
    include_replies: bool = typer.Option(
        True,
        "--include-replies/--no-replies",
        help="Include tweets that are replies to other users.",
    ),
) -> None:
    """Collect tweets from an X/Twitter account and store them in the local cache."""
    username = username.lstrip("@").lower()

    # Validate collector choice
    if collector not in ("api", "scrape", "archive"):
        console.print(
            f"[red]Unknown collector: {collector!r}. Choose 'api', 'scrape', or 'archive'.[/red]"
        )
        raise typer.Exit(1)

    if collector == "archive" and file is None:
        console.print(
            "[red]The --file option is required when using --collector=archive.[/red]"
        )
        raise typer.Exit(1)

    if collector == "archive" and file is not None and not file.exists():
        console.print(f"[red]Archive file not found: {file}[/red]")
        raise typer.Exit(1)

    console.print(
        f"\n[bold]Collecting tweets for @{username}[/bold] "
        f"(count={count}, collector={collector}, replies={'yes' if include_replies else 'no'})\n"
    )

    try:
        tweets = asyncio.run(_run_collection(username, count, collector, file, include_replies))
    except PermissionError as exc:
        console.print(f"[red]Authentication error: {exc}[/red]")
        raise typer.Exit(1)
    except LookupError as exc:
        console.print(f"[red]User not found: {exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Collection failed: {exc}[/red]")
        raise typer.Exit(1)

    if not tweets:
        console.print("[yellow]No tweets were collected.[/yellow]")
        raise typer.Exit(1)

    # Store in cache
    _save_to_cache(username, tweets)

    # Display summary
    _display_summary(username, tweets)


async def _run_collection(
    username: str,
    count: int,
    collector_type: str,
    file: Path | None,
    include_replies: bool,
) -> list[dict]:
    """Run the async collection process."""
    from postlikeme.models.config import AppConfig
    from postlikeme.storage.config_store import ConfigStore

    config_store = ConfigStore(AppConfig().data_dir)
    config = config_store.load()

    if collector_type == "api":
        from postlikeme.collectors.tweepy_collector import TweepyCollector

        bearer_token = config.x_bearer_token
        if not bearer_token:
            raise PermissionError(
                "X/Twitter Bearer Token is required. Set the X_BEARER_TOKEN "
                "environment variable or run 'postlikeme config set x_bearer_token <token>'."
            )
        collector_instance = TweepyCollector(bearer_token=bearer_token)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Collecting tweets for @{username}...", total=None
            )
            raw_tweets = await collector_instance.collect_user_tweets(
                username=username,
                count=count,
                include_replies=include_replies,
            )
            progress.update(task, completed=True)

        return [t.model_dump(mode="json") for t in raw_tweets]

    elif collector_type == "archive":
        return _load_archive(file, count, include_replies)  # type: ignore[arg-type]

    elif collector_type == "scrape":
        console.print(
            "[yellow]Scraping backend is not yet fully implemented. "
            "Please use 'api' or 'archive'.[/yellow]"
        )
        raise typer.Exit(1)

    return []


def _load_archive(file: Path, count: int, include_replies: bool) -> list[dict]:
    """Load tweets from a Twitter archive JSON file."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading archive file...", total=None)

        data = json.loads(file.read_text(encoding="utf-8"))

        # Twitter archive format: list of tweet objects or {"tweet": {...}} wrappers
        tweets_raw: list[dict] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    tweet_data = item.get("tweet", item)
                    tweets_raw.append(tweet_data)
        elif isinstance(data, dict) and "tweets" in data:
            for item in data["tweets"]:
                tweet_data = item.get("tweet", item) if isinstance(item, dict) else item
                tweets_raw.append(tweet_data)

        # Normalize to RawTweet-compatible format
        from datetime import datetime

        normalized: list[dict] = []
        for t in tweets_raw:
            tweet_text = t.get("full_text", t.get("text", ""))
            is_reply = bool(t.get("in_reply_to_user_id") or t.get("in_reply_to_status_id"))

            if not include_replies and is_reply:
                continue

            # Parse created_at
            created_str = t.get("created_at", "")
            try:
                created_at = datetime.strptime(
                    created_str, "%a %b %d %H:%M:%S %z %Y"
                ).isoformat()
            except (ValueError, TypeError):
                created_at = datetime.utcnow().isoformat()

            normalized.append({
                "id": str(t.get("id", t.get("id_str", ""))),
                "text": tweet_text,
                "created_at": created_at,
                "is_reply": is_reply,
                "reply_to_user": t.get("in_reply_to_screen_name"),
                "reply_to_tweet_id": str(t["in_reply_to_status_id"])
                if t.get("in_reply_to_status_id")
                else None,
                "is_retweet": tweet_text.startswith("RT @"),
                "is_quote_tweet": bool(t.get("quoted_status")),
                "quoted_text": t.get("quoted_status", {}).get("full_text")
                if isinstance(t.get("quoted_status"), dict)
                else None,
                "hashtags": [
                    ht.get("text", "")
                    for ht in t.get("entities", {}).get("hashtags", [])
                ],
                "mentions": [
                    m.get("screen_name", "")
                    for m in t.get("entities", {}).get("user_mentions", [])
                ],
                "urls": [
                    u.get("expanded_url", u.get("url", ""))
                    for u in t.get("entities", {}).get("urls", [])
                ],
                "media_types": [],
                "like_count": int(t.get("favorite_count", 0)),
                "retweet_count": int(t.get("retweet_count", 0)),
                "reply_count": 0,
                "language": t.get("lang"),
            })

            if len(normalized) >= count:
                break

        progress.update(task, completed=True)

    return normalized


def _save_to_cache(username: str, tweets: list[dict]) -> None:
    """Save collected tweets to the local cache directory."""
    from postlikeme.models.config import AppConfig

    config = AppConfig()
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"{username}_tweets.json"
    cache_file.write_text(
        json.dumps(tweets, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    console.print(f"[dim]Tweets cached at: {cache_file}[/dim]")


def _display_summary(username: str, tweets: list[dict]) -> None:
    """Display a summary table of collected tweets."""
    total = len(tweets)
    replies = sum(1 for t in tweets if t.get("is_reply", False))
    retweets = sum(1 for t in tweets if t.get("is_retweet", False))
    originals = total - replies - retweets

    # Date range
    dates = []
    for t in tweets:
        created = t.get("created_at", "")
        if created:
            dates.append(str(created)[:10])

    date_range = "N/A"
    if dates:
        dates.sort()
        date_range = f"{dates[0]} to {dates[-1]}"

    table = Table(title=f"Collection Summary for @{username}", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total tweets", str(total))
    table.add_row("Original tweets", str(originals))
    table.add_row("Replies", str(replies))
    table.add_row("Retweets", str(retweets))
    table.add_row("Date range", date_range)

    console.print()
    console.print(table)
    console.print()
