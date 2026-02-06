"""Tests for the CLI application entry point."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from postlikeme.cli.app import app


runner = CliRunner()


class TestCLI:
    """Verify CLI application loads and basic commands work."""

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "postlikeme" in result.output.lower() or "analyze" in result.output.lower()

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        # no_args_is_help=True should show help text
        assert "Usage" in result.output or "analyze" in result.output.lower()

    def test_collect_help(self):
        result = runner.invoke(app, ["collect", "--help"])
        assert result.exit_code == 0
        assert "username" in result.output.lower() or "count" in result.output.lower()

    def test_analyze_help(self):
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_generate_help(self):
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0

    def test_profile_help(self):
        result = runner.invoke(app, ["profile", "--help"])
        assert result.exit_code == 0

    def test_config_help(self):
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
