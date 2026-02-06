"""API router for tweet analysis and voice profile building."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException

from postlikeme.api.schemas import AnalyzeRequest, AnalyzeResponse
from postlikeme.models.config import AppConfig
from postlikeme.models.raw_tweet import RawTweet, UserProfile
from postlikeme.storage.config_store import ConfigStore
from postlikeme.storage.profile_store import ProfileStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze/{username}", response_model=AnalyzeResponse)
async def analyze_account(username: str, request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyse collected tweets and build a voice profile.

    Requires tweets to have been collected first via the /collect endpoint
    or the CLI. Uses the full analysis pipeline including structural,
    topic, personality, and reply analysis.
    """
    username = username.lstrip("@").lower()

    # Load config
    config_store = ConfigStore(AppConfig().data_dir)
    config = config_store.load()
    config.ensure_dirs()

    profile_store = ProfileStore(config.data_dir)

    # Check for existing profile
    if not request.rebuild:
        existing = profile_store.load_profile(username)
        if existing is not None:
            return AnalyzeResponse(
                username=username,
                tweet_count_analyzed=existing.tweet_count_analyzed,
                topics_found=len(existing.topic_profile.topics_ranked),
                message=f"Profile for @{username} already exists "
                f"(built {existing.profile_built_at.strftime('%Y-%m-%d %H:%M UTC')}). "
                "Set rebuild=true to regenerate.",
            )

    # Load cached tweets
    cache_file = config.cache_dir / f"{username}_tweets.json"
    if not cache_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No cached tweets found for @{username}. "
            f"Collect tweets first via POST /api/collect/{username}.",
        )

    try:
        tweets_data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read cached tweets for @{username}: {exc}",
        ) from exc

    if not tweets_data:
        raise HTTPException(
            status_code=400,
            detail=f"Cache file for @{username} is empty. Re-collect tweets first.",
        )

    # Convert cached dicts to RawTweet objects
    try:
        raw_tweets = [RawTweet.model_validate(t) for t in tweets_data]
    except Exception as exc:
        logger.warning(
            "Failed to parse some cached tweets as RawTweet; falling back to dict-based analysis: %s",
            exc,
        )
        # If RawTweet validation fails (e.g., old cache format), try to proceed
        # by constructing minimal RawTweet objects
        raw_tweets = []
        from datetime import datetime, timezone

        for t in tweets_data:
            try:
                created_at_str = t.get("created_at", "")
                if isinstance(created_at_str, str):
                    try:
                        created_at = datetime.fromisoformat(
                            created_at_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        created_at = datetime.now(timezone.utc)
                else:
                    created_at = datetime.now(timezone.utc)

                raw_tweets.append(
                    RawTweet(
                        id=str(t.get("id", "")),
                        text=t.get("text", ""),
                        created_at=created_at,
                        is_reply=t.get("is_reply", False),
                        reply_to_user=t.get("reply_to_user"),
                        reply_to_tweet_id=t.get("reply_to_tweet_id"),
                        is_retweet=t.get("is_retweet", False),
                        is_quote_tweet=t.get("is_quote_tweet", False),
                        quoted_text=t.get("quoted_text"),
                        hashtags=t.get("hashtags", []),
                        mentions=t.get("mentions", []),
                        urls=t.get("urls", []),
                        media_types=t.get("media_types", []),
                        like_count=int(t.get("like_count", 0)),
                        retweet_count=int(t.get("retweet_count", 0)),
                        reply_count=int(t.get("reply_count", 0)),
                        language=t.get("language"),
                    )
                )
            except Exception:
                continue

    if not raw_tweets:
        raise HTTPException(
            status_code=400,
            detail=f"No valid tweets could be parsed from cache for @{username}.",
        )

    # Build a UserProfile from cached data (we may not have full profile info)
    user_profile = UserProfile(
        username=username,
        display_name=username,
        bio="",
    )

    # Run the analysis pipeline
    try:
        from postlikeme.analysis.pipeline import AnalysisPipeline

        pipeline = AnalysisPipeline(config)
        profile = await pipeline.analyze(raw_tweets, user_profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analysis failed for @%s", username)
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {exc}"
        ) from exc

    # Save the profile
    profile_store.save_profile(profile)

    return AnalyzeResponse(
        username=username,
        tweet_count_analyzed=profile.tweet_count_analyzed,
        topics_found=len(profile.topic_profile.topics_ranked),
        message=f"Successfully analysed {profile.tweet_count_analyzed} tweets for @{username}.",
    )
