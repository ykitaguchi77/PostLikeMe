"""Abstract LLM client with Anthropic and OpenAI backend implementations.

Provides a unified async interface for text generation so the rest of the
system stays agnostic about the underlying model provider.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from postlikeme.models.config import AppConfig

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM text generation."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.85,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text from a system + user prompt pair.

        Parameters
        ----------
        system_prompt:
            The system-level instruction that shapes the model's behaviour.
        user_prompt:
            The user-facing prompt containing examples and the generation request.
        temperature:
            Sampling temperature (0.0 = deterministic, higher = more creative).
        max_tokens:
            Maximum number of tokens in the generated response.

        Returns
        -------
        str
            The generated text content.
        """
        ...


class AnthropicClient(BaseLLMClient):
    """LLM client using the Anthropic Messages API.

    Parameters
    ----------
    api_key:
        Anthropic API key.
    model:
        Model identifier (e.g. ``"claude-sonnet-4-20250514"``).
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.85,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text using the Anthropic Messages API."""
        logger.debug(
            "Anthropic generate: model=%s, temperature=%.2f, max_tokens=%d",
            self.model,
            temperature,
            max_tokens,
        )
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class OpenAIClient(BaseLLMClient):
    """LLM client using the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key:
        OpenAI API key.
    model:
        Model identifier (e.g. ``"gpt-4o"``).
    """

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.85,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text using the OpenAI Chat Completions API."""
        logger.debug(
            "OpenAI generate: model=%s, temperature=%.2f, max_tokens=%d",
            self.model,
            temperature,
            max_tokens,
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0]
        return choice.message.content or ""


def create_llm_client(config: AppConfig) -> BaseLLMClient:
    """Factory function to create the appropriate LLM client based on config.

    Reads ``config.default_llm_provider`` and ``config.default_llm_model`` to
    determine which backend to instantiate, and resolves the corresponding API
    key from the config (which itself falls back to environment variables).

    Parameters
    ----------
    config:
        The application configuration.

    Returns
    -------
    BaseLLMClient
        A configured LLM client ready for use.

    Raises
    ------
    ValueError
        If the required API key is not set or the provider is unknown.
    """
    provider = config.default_llm_provider
    model = config.default_llm_model

    if provider == "anthropic":
        api_key = config.anthropic_api_key
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Set the ANTHROPIC_API_KEY environment "
                "variable or run 'postlikeme config set anthropic_api_key <key>'."
            )
        return AnthropicClient(api_key=api_key, model=model)

    elif provider == "openai":
        api_key = config.openai_api_key
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set the OPENAI_API_KEY environment "
                "variable or run 'postlikeme config set openai_api_key <key>'."
            )
        return OpenAIClient(api_key=api_key, model=model)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Supported providers: 'anthropic', 'openai'."
        )
