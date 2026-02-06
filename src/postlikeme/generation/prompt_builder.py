"""Prompt construction from Jinja2 templates and VoiceProfile data.

Loads templates from the ``postlikeme/templates/`` directory and renders
them with profile data to produce system and user prompts for the LLM.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jinja2

from postlikeme.models.voice_profile import VoiceProfile
from postlikeme.utils.constants import MAX_TWEET_LENGTH

logger = logging.getLogger(__name__)

# Default templates directory relative to this package
_DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


class PromptBuilder:
    """Builds LLM prompts from Jinja2 templates and VoiceProfile data.

    Parameters
    ----------
    template_dir:
        Path to the directory containing ``.j2`` template files.
        Defaults to the ``postlikeme/templates/`` package directory.
    """

    def __init__(self, template_dir: Path | None = None) -> None:
        self._template_dir = template_dir or _DEFAULT_TEMPLATE_DIR

        if not self._template_dir.is_dir():
            raise FileNotFoundError(
                f"Template directory not found: {self._template_dir}. "
                f"Ensure the PostLikeMe package is installed correctly."
            )

        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self._template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.Undefined,
            keep_trailing_newline=False,
        )

        # Register custom filters
        self._env.filters["regex_replace"] = self._regex_replace_filter

        logger.debug("PromptBuilder initialized with templates from %s", self._template_dir)

    @staticmethod
    def _regex_replace_filter(value: str, pattern: str, replacement: str) -> str:
        """Jinja2 filter for regex replacement."""
        import re

        return re.sub(pattern, replacement, value)

    def build_system_prompt(
        self,
        profile: VoiceProfile,
        max_length: int = MAX_TWEET_LENGTH,
    ) -> str:
        """Render the system prompt template with the given voice profile.

        Parameters
        ----------
        profile:
            The voice profile to inject into the template.
        max_length:
            Maximum tweet character length (passed to template as ``max_length``).

        Returns
        -------
        str
            The fully rendered system prompt.
        """
        template = self._env.get_template("system_prompt.j2")
        rendered = template.render(profile=profile, max_length=max_length)
        # Clean up excessive blank lines from conditional template blocks
        lines = rendered.split("\n")
        cleaned_lines: list[str] = []
        prev_blank = False
        for line in lines:
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue
            cleaned_lines.append(line)
            prev_blank = is_blank
        return "\n".join(cleaned_lines).strip()

    def build_tweet_prompt(
        self,
        profile: VoiceProfile,
        topic: str | None = None,
    ) -> str:
        """Render the tweet generation prompt template.

        Parameters
        ----------
        profile:
            The voice profile providing example tweets.
        topic:
            Optional topic to guide the generated tweet. If ``None``, the
            model picks a topic naturally.

        Returns
        -------
        str
            The fully rendered user prompt for tweet generation.
        """
        template = self._env.get_template("tweet_prompt.j2")
        return template.render(profile=profile, topic=topic).strip()

    def build_reply_prompt(
        self,
        profile: VoiceProfile,
        target_tweet: str,
    ) -> str:
        """Render the reply generation prompt template.

        Parameters
        ----------
        profile:
            The voice profile providing example reply pairs.
        target_tweet:
            The tweet text to generate a reply to.

        Returns
        -------
        str
            The fully rendered user prompt for reply generation.
        """
        template = self._env.get_template("reply_prompt.j2")
        return template.render(profile=profile, target_tweet=target_tweet).strip()
