"""Tweet generation engine.

Orchestrates the LLM client, prompt builder, and post-processor to
produce style-matched tweet candidates from a VoiceProfile.
"""

from __future__ import annotations

import asyncio
import logging

from postlikeme.generation.llm_client import BaseLLMClient
from postlikeme.generation.post_processor import PostProcessor
from postlikeme.generation.prompt_builder import PromptBuilder
from postlikeme.models.voice_profile import VoiceProfile
from postlikeme.utils.constants import MAX_TWEET_LENGTH

logger = logging.getLogger(__name__)


class TweetGenerator:
    """Generates style-matched tweets using an LLM.

    Parameters
    ----------
    llm_client:
        The LLM backend for text generation.
    prompt_builder:
        Renders Jinja2 templates into prompts.
    post_processor:
        Cleans and scores generated output. If ``None``, a default
        :class:`PostProcessor` is created.
    max_length:
        Maximum tweet character length for validation.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_builder: PromptBuilder,
        post_processor: PostProcessor | None = None,
        max_length: int = MAX_TWEET_LENGTH,
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._post_processor = post_processor or PostProcessor()
        self._max_length = max_length

    async def generate(
        self,
        profile: VoiceProfile,
        topic: str | None = None,
        count: int = 5,
        temperature: float = 0.85,
    ) -> list[str]:
        """Generate tweet candidates matching the voice profile.

        Makes ``count`` independent LLM calls, post-processes each result,
        and returns only valid candidates (non-empty, within length limit).

        Parameters
        ----------
        profile:
            The voice profile to emulate.
        topic:
            Optional topic to guide the generation.
        count:
            Number of tweet candidates to generate.
        temperature:
            LLM sampling temperature. Higher values produce more variety.

        Returns
        -------
        list[str]
            Valid generated tweet texts (may be fewer than *count* if some
            candidates fail validation).
        """
        system_prompt = self._prompt_builder.build_system_prompt(
            profile, max_length=self._max_length
        )
        user_prompt = self._prompt_builder.build_tweet_prompt(profile, topic)

        logger.info(
            "Generating %d tweet candidates for @%s (topic=%s, temp=%.2f)",
            count,
            profile.username,
            topic or "auto",
            temperature,
        )

        # Generate all candidates concurrently for speed
        tasks = [
            self._generate_single(system_prompt, user_prompt, temperature)
            for _ in range(count)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[str] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Tweet generation attempt %d failed: %s", i + 1, result)
                continue
            if result:
                candidates.append(result)

        logger.info(
            "Generated %d valid candidates out of %d attempts for @%s",
            len(candidates),
            count,
            profile.username,
        )
        return candidates

    async def generate_ranked(
        self,
        profile: VoiceProfile,
        topic: str | None = None,
        count: int = 5,
        temperature: float = 0.85,
    ) -> list[tuple[str, float]]:
        """Generate and rank tweet candidates by style adherence.

        Parameters
        ----------
        profile:
            The voice profile to emulate.
        topic:
            Optional topic to guide the generation.
        count:
            Number of tweet candidates to generate.
        temperature:
            LLM sampling temperature.

        Returns
        -------
        list[tuple[str, float]]
            Candidates paired with style adherence scores, sorted
            descending (best match first).
        """
        candidates = await self.generate(profile, topic, count, temperature)
        return self._post_processor.rank_candidates(candidates, profile)

    async def _generate_single(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str | None:
        """Generate a single tweet candidate and post-process it.

        Returns
        -------
        str | None
            The cleaned tweet text, or ``None`` if it fails validation.
        """
        raw_text = await self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=512,
        )

        cleaned = self._post_processor.clean_llm_output(raw_text)

        if not cleaned:
            logger.debug("LLM returned empty output after cleaning.")
            return None

        if not self._post_processor.validate_length(cleaned, self._max_length):
            logger.debug(
                "Generated tweet exceeds %d chars (%d chars): %s...",
                self._max_length,
                len(cleaned),
                cleaned[:80],
            )
            # Try truncation as a last resort -- only if slightly over
            if len(cleaned) <= self._max_length + 20:
                # Find last sentence boundary before limit
                truncated = self._smart_truncate(cleaned, self._max_length)
                if truncated and self._post_processor.validate_length(
                    truncated, self._max_length
                ):
                    return truncated
            return None

        return cleaned

    @staticmethod
    def _smart_truncate(text: str, max_length: int) -> str:
        """Truncate text at a natural boundary before max_length.

        Tries to cut at the last sentence-ending punctuation, then at the
        last space, then hard-truncates as a final fallback.
        """
        if len(text) <= max_length:
            return text

        # Try to find the last sentence boundary within limit
        truncation_zone = text[:max_length]

        for punct in [".", "!", "?"]:
            last_idx = truncation_zone.rfind(punct)
            if last_idx > max_length * 0.5:
                return text[: last_idx + 1].strip()

        # Fall back to last space
        last_space = truncation_zone.rfind(" ")
        if last_space > max_length * 0.5:
            return text[:last_space].strip()

        # Hard truncate (rare)
        return text[:max_length].strip()
