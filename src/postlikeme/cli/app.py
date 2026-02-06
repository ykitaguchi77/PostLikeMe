"""Main Typer CLI application for PostLikeMe.

Defines the top-level ``postlikeme`` command group and registers all
sub-commands (collect, analyze, generate, profile, config).
"""

from __future__ import annotations

import typer

from postlikeme.cli.analyze_cmd import analyze
from postlikeme.cli.collect_cmd import collect
from postlikeme.cli.config_cmd import config_app
from postlikeme.cli.generate_cmd import generate_app
from postlikeme.cli.profile_cmd import profile_app

app = typer.Typer(
    name="postlikeme",
    help="Analyze X/Twitter accounts and generate style-matched tweets.",
    no_args_is_help=True,
)

# Register top-level commands
app.command()(collect)
app.command()(analyze)

# Register sub-command groups
app.add_typer(generate_app, name="generate")
app.add_typer(profile_app, name="profile")
app.add_typer(config_app, name="config")


@app.callback()
def callback() -> None:
    """PostLikeMe - Analyze X/Twitter accounts and generate style-matched tweets."""
    pass


def main() -> None:
    """Entry point for the CLI."""
    app()
