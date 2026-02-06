"""API router for tweet collection."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException

from postlikeme.api.schemas import CollectRequest, CollectResponse
from postlikeme.models.config import AppConfig
from postlikeme.storage.config_store import ConfigStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/collect/{username}", response_model=CollectResponse)
async def collect_tweets(username: str, request: CollectRequest) -> CollectResponse:
    """Collect tweets from an X/Twitter account.

    Uses the configured collection backend (API, scrape, or archive)
    to fetch tweets and store them in the local cache.
    """
    username = username.lstrip("@").lower()

    # Load config
    config_store = ConfigStore(AppConfig().data_dir)
    config = config_store.load()
    config.ensure_dirs()

    collector_type = request.collector
    if collector_type not in ("api", "scrape", "archive"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown collector: {collector_type!r}. Choose 'api', 'scrape', or 'archive'.",
        )

    try:
        if collector_type == "api":
            bearer_token = config.x_bearer_token
            if not bearer_token:
                raise HTTPException(
                    status_code=400,
                    detail="X/Twitter Bearer Token is required. Set the X_BEARER_TOKEN "
                    "environment variable or configure it via 'postlikeme config set x_bearer_token <token>'.",
                )

            from postlikeme.collectors.tweepy_collector import TweepyCollector

            collector_instance = TweepyCollector(bearer_token=bearer_token)
            raw_tweets = await collector_instance.collect_user_tweets(
                username=username,
                count=request.count,
                include_replies=request.include_replies,
            )

            # Serialize and cache
            tweets_data = [t.model_dump(mode="json") for t in raw_tweets]

        elif collector_type == "scrape":
            raise HTTPException(
                status_code=501,
                detail="Scraping backend is not yet fully implemented. Use 'api' or 'archive'.",
            )

        elif collector_type == "archive":
            raise HTTPException(
                status_code=400,
                detail="Archive collection is not supported via the API. Use the CLI instead.",
            )
        else:
            tweets_data = []

    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=f"Authentication error: {exc}") from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=f"User not found: {exc}") from exc
    except Exception as exc:
        logger.exception("Collection failed for @%s", username)
        raise HTTPException(status_code=500, detail=f"Collection failed: {exc}") from exc

    if not tweets_data:
        raise HTTPException(status_code=404, detail=f"No tweets found for @{username}.")

    # Save to cache
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{username}_tweets.json"
    cache_file.write_text(
        json.dumps(tweets_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return CollectResponse(
        username=username,
        tweet_count=len(tweets_data),
        message=f"Successfully collected {len(tweets_data)} tweets for @{username}.",
    )
