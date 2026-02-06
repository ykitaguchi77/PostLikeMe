"""CLI commands for managing voice profiles.

Usage::

    postlikeme profile show <username>
    postlikeme profile list
    postlikeme profile export <username> [--format json|markdown]
    postlikeme profile delete <username>
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

profile_app = typer.Typer(
    name="profile",
    help="Manage voice profiles.",
)


def _get_profile_store():
    """Create and return a ProfileStore using the current config."""
    from postlikeme.models.config import AppConfig
    from postlikeme.storage.config_store import ConfigStore
    from postlikeme.storage.profile_store import ProfileStore

    config_store = ConfigStore(AppConfig().data_dir)
    config = config_store.load()
    config.ensure_dirs()
    return ProfileStore(config.data_dir)


@profile_app.command("show")
def show(
    username: str = typer.Argument(
        ...,
        help="Username of the profile to display (without @ prefix).",
    ),
) -> None:
    """Display a detailed view of a voice profile."""
    username = username.lstrip("@").lower()
    profile_store = _get_profile_store()

    profile = profile_store.load_profile(username)
    if profile is None:
        console.print(
            f"[red]No profile found for @{username}. "
            f"Run 'postlikeme collect {username}' and "
            f"'postlikeme analyze {username}' first.[/red]"
        )
        raise typer.Exit(1)

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]@{profile.username}[/bold]"
            + (f" ({profile.display_name})" if profile.display_name else "")
            + (f"\n[dim]{profile.bio}[/dim]" if profile.bio else ""),
            title="Voice Profile",
            border_style="blue",
        )
    )

    # Metadata
    built_at = (
        profile.profile_built_at.strftime("%Y-%m-%d %H:%M UTC")
        if profile.profile_built_at
        else "N/A"
    )
    date_start = (
        profile.date_range_start.strftime("%Y-%m-%d")
        if profile.date_range_start
        else "N/A"
    )
    date_end = (
        profile.date_range_end.strftime("%Y-%m-%d")
        if profile.date_range_end
        else "N/A"
    )

    meta_table = Table(title="Metadata", show_header=False, border_style="dim")
    meta_table.add_column("Key", style="bold")
    meta_table.add_column("Value")
    meta_table.add_row("Tweets analysed", str(profile.tweet_count_analyzed))
    meta_table.add_row("Profile built", built_at)
    meta_table.add_row("Date range", f"{date_start} to {date_end}")
    meta_table.add_row("Schema version", profile.schema_version)
    console.print(meta_table)
    console.print()

    # Personality Panel
    pp = profile.personality_profile
    b5 = pp.big_five
    personality_lines = [
        f"[bold]Formality:[/bold] {pp.formality_score:.2f}",
        f"[bold]Tone:[/bold] {pp.tone}",
        f"[bold]Humor style:[/bold] {pp.humor_style}",
        f"[bold]Assertiveness:[/bold] {pp.assertiveness:.2f}",
        "",
        "[bold]Big Five Personality Traits:[/bold]",
        f"  Openness:          {b5.openness:.2f}",
        f"  Conscientiousness: {b5.conscientiousness:.2f}",
        f"  Extraversion:      {b5.extraversion:.2f}",
        f"  Agreeableness:     {b5.agreeableness:.2f}",
        f"  Neuroticism:       {b5.neuroticism:.2f}",
    ]
    if pp.rhetorical_devices:
        personality_lines.append("")
        personality_lines.append(
            f"[bold]Rhetorical devices:[/bold] {', '.join(pp.rhetorical_devices)}"
        )
    if pp.catchphrases:
        personality_lines.append(
            f"[bold]Catchphrases:[/bold] {', '.join(repr(c) for c in pp.catchphrases[:5])}"
        )

    console.print(
        Panel(
            "\n".join(personality_lines),
            title="Personality",
            border_style="magenta",
        )
    )

    # Topics Panel
    tp = profile.topic_profile
    if tp.topics_ranked:
        topic_lines = []
        for t in tp.topics_ranked[:10]:
            kw = ", ".join(t.keywords[:5]) if t.keywords else ""
            topic_lines.append(
                f"[bold]{t.name}[/bold] (importance: {t.importance:.2f})"
                + (f" -- {kw}" if kw else "")
            )

        # Add opinion map entries
        if tp.opinion_map:
            topic_lines.append("")
            topic_lines.append("[bold]Sentiment by topic:[/bold]")
            for op in tp.opinion_map[:10]:
                topic_lines.append(
                    f"  {op.topic}: {op.label} (mean={op.sentiment_mean:+.2f})"
                )

        console.print(
            Panel(
                "\n".join(topic_lines),
                title="Topics",
                border_style="green",
            )
        )

    # Style Panel
    ss = profile.structural_style
    style_lines = [
        f"[bold]Avg tweet length:[/bold] {ss.avg_tweet_length:.0f} chars",
        f"[bold]Avg sentence length:[/bold] {ss.avg_sentence_length:.1f} tokens",
        f"[bold]Fragment ratio:[/bold] {ss.fragment_ratio:.1%}",
        f"[bold]Question ratio:[/bold] {ss.question_ratio:.1%}",
        f"[bold]Exclamation ratio:[/bold] {ss.exclamation_ratio:.1%}",
    ]
    if ss.readability_scores:
        rs = ss.readability_scores
        style_lines.append("")
        style_lines.append("[bold]Readability:[/bold]")
        style_lines.append(f"  Flesch Reading Ease: {rs.flesch_reading_ease:.1f}")
        style_lines.append(f"  Gunning Fog:         {rs.gunning_fog:.1f}")

    if ss.vocabulary_richness:
        vr = ss.vocabulary_richness
        style_lines.append("")
        style_lines.append("[bold]Vocabulary richness:[/bold]")
        if "ttr" in vr:
            style_lines.append(f"  TTR:  {vr['ttr']:.4f}")
        if "mtld" in vr:
            style_lines.append(f"  MTLD: {vr['mtld']:.1f}")

    console.print(
        Panel(
            "\n".join(style_lines),
            title="Structural Style",
            border_style="cyan",
        )
    )

    # Emoji Usage Panel
    ep = profile.emoji_profile
    emoji_lines = [
        f"[bold]Emojis per tweet:[/bold] {ep.usage_rate:.2f}",
        f"[bold]Tweets with emojis:[/bold] {ep.emoji_tweet_ratio:.1%}",
        f"[bold]Emoji diversity:[/bold] {ep.emoji_diversity:.2f}",
        f"[bold]Emoji sentiment:[/bold] {ep.emoji_sentiment:+.2f}",
    ]
    if ep.top_emojis:
        top_e = " ".join(f"{e}({c})" for e, c in ep.top_emojis[:10])
        emoji_lines.append(f"[bold]Top emojis:[/bold] {top_e}")
    if ep.positional_tendency:
        pos = ep.positional_tendency
        emoji_lines.append(
            f"[bold]Position:[/bold] leading={pos.get('leading', 0):.0%} "
            f"inline={pos.get('inline', 0):.0%} "
            f"trailing={pos.get('trailing', 0):.0%}"
        )

    console.print(
        Panel(
            "\n".join(emoji_lines),
            title="Emoji Usage",
            border_style="yellow",
        )
    )

    # Reply Style Panel
    rp = profile.reply_style
    reply_lines = [
        f"[bold]Avg reply length:[/bold] {rp.avg_reply_length:.0f} chars",
        f"[bold]Length ratio (reply/original):[/bold] {rp.length_ratio:.2f}",
        f"[bold]Tone vs originals:[/bold] {rp.tone_vs_originals}",
        f"[bold]Engagement type:[/bold] {rp.engagement_type}",
        f"[bold]Sentiment diff:[/bold] {rp.sentiment_diff:+.3f}",
    ]
    if rp.opening_patterns:
        reply_lines.append("[bold]Opening patterns:[/bold]")
        for pattern, ratio in rp.opening_patterns.items():
            reply_lines.append(f"  {pattern}: {ratio:.1%}")

    console.print(
        Panel(
            "\n".join(reply_lines),
            title="Reply Style",
            border_style="red",
        )
    )

    # Examples summary
    ex = profile.example_tweets
    console.print(
        f"\n[dim]Examples: {len(ex.originals)} original tweets, "
        f"{len(ex.replies)} reply pairs[/dim]\n"
    )


@profile_app.command("list")
def list_profiles() -> None:
    """List all stored voice profiles."""
    profile_store = _get_profile_store()
    usernames = profile_store.list_profiles()

    if not usernames:
        console.print("[yellow]No profiles found. Run 'postlikeme analyze <username>' to create one.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Voice Profiles", show_lines=False)
    table.add_column("Username", style="bold")
    table.add_column("Display Name")
    table.add_column("Built At")
    table.add_column("Tweets Analysed", justify="right")

    for uname in usernames:
        profile = profile_store.load_profile(uname)
        if profile is None:
            continue

        built_at = (
            profile.profile_built_at.strftime("%Y-%m-%d %H:%M")
            if profile.profile_built_at
            else "N/A"
        )
        table.add_row(
            f"@{profile.username}",
            profile.display_name or "-",
            built_at,
            str(profile.tweet_count_analyzed),
        )

    console.print()
    console.print(table)
    console.print()


@profile_app.command("export")
def export(
    username: str = typer.Argument(
        ...,
        help="Username of the profile to export.",
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Export format: 'json' or 'markdown'.",
    ),
) -> None:
    """Export a voice profile to stdout in the specified format."""
    username = username.lstrip("@").lower()
    profile_store = _get_profile_store()

    try:
        output = profile_store.export_profile(username, format=format)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(output)


@profile_app.command("delete")
def delete(
    username: str = typer.Argument(
        ...,
        help="Username of the profile to delete.",
    ),
) -> None:
    """Delete a stored voice profile."""
    username = username.lstrip("@").lower()
    profile_store = _get_profile_store()

    # Check if profile exists first
    profile = profile_store.load_profile(username)
    if profile is None:
        console.print(f"[yellow]No profile found for @{username}.[/yellow]")
        raise typer.Exit(1)

    # Confirm deletion
    confirmed = typer.confirm(
        f"Are you sure you want to delete the profile for @{username}?"
    )
    if not confirmed:
        console.print("[dim]Deletion cancelled.[/dim]")
        raise typer.Exit(0)

    deleted = profile_store.delete_profile(username)
    if deleted:
        console.print(f"[green]Profile for @{username} has been deleted.[/green]")
    else:
        console.print(f"[red]Failed to delete profile for @{username}.[/red]")
        raise typer.Exit(1)
