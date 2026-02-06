"""API router for tweet and reply generation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from postlikeme.api.schemas import (
    GenerateCandidate,
    GenerateReplyRequest,
    GenerateResponse,
    GenerateTweetRequest,
)
from postlikeme.models.config import AppConfig
from postlikeme.storage.config_store import ConfigStore
from postlikeme.storage.profile_store import ProfileStore

logger = logging.getLogger(__name__)

router = APIRouter()


def _load_profile_or_404(username: str):
    """Load a voice profile, raising HTTP 404 if not found."""
    config_store = ConfigStore(AppConfig().data_dir)
    config = config_store.load()
    profile_store = ProfileStore(config.data_dir)

    profile = profile_store.load_profile(username)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for @{username}. "
            f"Run collection and analysis first.",
        )
    return profile, config


@router.post("/generate/tweet", response_model=GenerateResponse)
async def generate_tweet(request: GenerateTweetRequest) -> GenerateResponse:
    """Generate style-matched tweets for a profiled user.

    Requires a voice profile to exist for the specified username.
    Returns ranked candidates with style adherence scores.
    """
    username = request.username.lstrip("@").lower()

    profile, config = _load_profile_or_404(username)

    try:
        from postlikeme.generation.llm_client import create_llm_client
        from postlikeme.generation.post_processor import PostProcessor
        from postlikeme.generation.prompt_builder import PromptBuilder
        from postlikeme.generation.tweet_generator import TweetGenerator

        llm_client = create_llm_client(config)
        prompt_builder = PromptBuilder()
        post_processor = PostProcessor()
        generator = TweetGenerator(llm_client, prompt_builder, post_processor)

        ranked = await generator.generate_ranked(
            profile=profile,
            topic=request.topic,
            count=request.count,
            temperature=request.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Tweet generation failed for @%s", username)
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {exc}"
        ) from exc

    candidates = [
        GenerateCandidate(text=text, score=round(score, 4))
        for text, score in ranked
    ]

    return GenerateResponse(candidates=candidates)


@router.post("/generate/reply", response_model=GenerateResponse)
async def generate_reply(request: GenerateReplyRequest) -> GenerateResponse:
    """Generate style-matched replies for a profiled user.

    Requires a voice profile to exist for the specified username.
    Returns ranked reply candidates with style adherence scores.
    """
    username = request.username.lstrip("@").lower()

    if not request.target_tweet.strip():
        raise HTTPException(
            status_code=400,
            detail="target_tweet must not be empty.",
        )

    profile, config = _load_profile_or_404(username)

    try:
        from postlikeme.generation.llm_client import create_llm_client
        from postlikeme.generation.post_processor import PostProcessor
        from postlikeme.generation.prompt_builder import PromptBuilder
        from postlikeme.generation.reply_generator import ReplyGenerator

        llm_client = create_llm_client(config)
        prompt_builder = PromptBuilder()
        post_processor = PostProcessor()
        generator = ReplyGenerator(llm_client, prompt_builder, post_processor)

        ranked = await generator.generate_ranked(
            profile=profile,
            target_tweet=request.target_tweet,
            count=request.count,
            temperature=request.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Reply generation failed for @%s", username)
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {exc}"
        ) from exc

    candidates = [
        GenerateCandidate(text=text, score=round(score, 4))
        for text, score in ranked
    ]

    return GenerateResponse(candidates=candidates)
