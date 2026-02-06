"""Tweepy-based collector using the X/Twitter API v2.

Implements :class:`BaseCollector` using the official ``tweepy`` library with
Bearer Token authentication (app-only auth).  Handles pagination, rate limiting,
and full entity parsing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import tweepy
from tweepy.errors import (
    BadRequest,
    Forbidden,
    TooManyRequests,
    TweepyException,
    Unauthorized,
)

from postlikeme.collectors.base import BaseCollector
from postlikeme.collectors.rate_limiter import RateLimiter
from postlikeme.models.raw_tweet import RawTweet, UserProfile

logger = logging.getLogger(__name__)

# Twitter API v2 rate limits for app-only auth:
# GET /2/users/:id/tweets  -> 900 requests / 15 min (1500 for user auth)
_USER_TIMELINE_MAX_REQUESTS = 900
_USER_TIMELINE_WINDOW_SECONDS = 15 * 60  # 15 minutes

# Fields to request from the Twitter API v2
_TWEET_FIELDS = [
    "id",
    "text",
    "created_at",
    "public_metrics",
    "referenced_tweets",
    "entities",
    "in_reply_to_user_id",
    "lang",
    "attachments",
]

_USER_FIELDS = [
    "id",
    "name",
    "username",
    "description",
    "public_metrics",
]

_EXPANSIONS = [
    "referenced_tweets.id",
    "in_reply_to_user_id",
    "attachments.media_keys",
]

_MEDIA_FIELDS = ["type"]


class TweepyCollector(BaseCollector):
    """Collect tweets using the official X/Twitter API v2 via tweepy.

    Parameters
    ----------
    bearer_token:
        X/Twitter API v2 Bearer Token for app-only authentication.
    """

    def __init__(self, bearer_token: str) -> None:
        if not bearer_token:
            raise ValueError(
                "A valid X/Twitter API v2 Bearer Token is required. "
                "Set the X_BEARER_TOKEN environment variable or pass it explicitly."
            )
        self._client = tweepy.Client(
            bearer_token=bearer_token,
            wait_on_rate_limit=False,  # We manage rate limiting ourselves
        )
        self._rate_limiter = RateLimiter(
            max_requests=_USER_TIMELINE_MAX_REQUESTS,
            window_seconds=_USER_TIMELINE_WINDOW_SECONDS,
        )

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    async def collect_user_tweets(
        self,
        username: str,
        count: int = 500,
        include_replies: bool = True,
    ) -> list[RawTweet]:
        """Collect up to *count* recent tweets for *username*.

        Uses paginated requests to the ``GET /2/users/:id/tweets`` endpoint.
        Retweets are included (they are marked with ``is_retweet=True``) so
        the preprocessor can filter them downstream.

        Parameters
        ----------
        username:
            Twitter handle without ``@``.
        count:
            Maximum number of tweets to retrieve.
        include_replies:
            If ``False``, tweets that are replies are excluded from results.

        Returns
        -------
        list[RawTweet]
            Tweets in reverse-chronological order.
        """
        user_id = await self._resolve_user_id(username)
        tweets: list[RawTweet] = []
        pagination_token: str | None = None
        remaining = count

        while remaining > 0:
            page_size = min(remaining, 100)  # API max per page is 100

            try:
                async with self._rate_limiter:
                    response = self._client.get_users_tweets(
                        id=user_id,
                        max_results=page_size,
                        pagination_token=pagination_token,
                        tweet_fields=_TWEET_FIELDS,
                        expansions=_EXPANSIONS,
                        media_fields=_MEDIA_FIELDS,
                        user_fields=["username"],
                    )
            except TooManyRequests:
                logger.warning(
                    "Rate limit hit while collecting tweets for @%s. "
                    "Returning %d tweets collected so far.",
                    username,
                    len(tweets),
                )
                break
            except Unauthorized as exc:
                raise PermissionError(
                    f"Authentication failed for Twitter API. "
                    f"Check your Bearer Token. Detail: {exc}"
                ) from exc
            except (BadRequest, Forbidden) as exc:
                raise LookupError(
                    f"Cannot access tweets for @{username}: {exc}"
                ) from exc
            except TweepyException as exc:
                logger.error(
                    "Tweepy error collecting tweets for @%s: %s", username, exc
                )
                break

            if response.data is None:
                logger.debug("No more tweets returned for @%s.", username)
                break

            # Build lookup maps from includes
            includes = response.includes or {}
            referenced_tweets_map = self._build_referenced_tweets_map(includes)
            media_map = self._build_media_map(includes)
            users_map = self._build_users_map(includes)

            for tweet_data in response.data:
                raw_tweet = self._parse_tweet(
                    tweet_data,
                    referenced_tweets_map=referenced_tweets_map,
                    media_map=media_map,
                    users_map=users_map,
                )

                if not include_replies and raw_tweet.is_reply:
                    continue

                tweets.append(raw_tweet)
                remaining -= 1
                if remaining <= 0:
                    break

            # Check for next page
            meta = response.meta or {}
            pagination_token = meta.get("next_token")
            if pagination_token is None:
                break

        logger.info("Collected %d tweets for @%s.", len(tweets), username)
        return tweets

    async def collect_tweet_by_id(self, tweet_id: str) -> RawTweet:
        """Fetch a single tweet by its ID.

        Parameters
        ----------
        tweet_id:
            Numeric tweet identifier as a string.

        Returns
        -------
        RawTweet

        Raises
        ------
        LookupError
            If the tweet does not exist or is not accessible.
        """
        try:
            async with self._rate_limiter:
                response = self._client.get_tweet(
                    id=tweet_id,
                    tweet_fields=_TWEET_FIELDS,
                    expansions=_EXPANSIONS,
                    media_fields=_MEDIA_FIELDS,
                    user_fields=["username"],
                )
        except (BadRequest, Forbidden, TweepyException) as exc:
            raise LookupError(
                f"Cannot retrieve tweet {tweet_id}: {exc}"
            ) from exc

        if response.data is None:
            raise LookupError(
                f"Tweet {tweet_id} does not exist or is not accessible."
            )

        includes = response.includes or {}
        referenced_tweets_map = self._build_referenced_tweets_map(includes)
        media_map = self._build_media_map(includes)
        users_map = self._build_users_map(includes)

        return self._parse_tweet(
            response.data,
            referenced_tweets_map=referenced_tweets_map,
            media_map=media_map,
            users_map=users_map,
        )

    async def get_user_profile(self, username: str) -> UserProfile:
        """Fetch public profile information for *username*.

        Parameters
        ----------
        username:
            Twitter handle without ``@``.

        Returns
        -------
        UserProfile

        Raises
        ------
        LookupError
            If the user does not exist or is suspended.
        """
        try:
            async with self._rate_limiter:
                response = self._client.get_user(
                    username=username,
                    user_fields=_USER_FIELDS,
                )
        except (BadRequest, Forbidden, TweepyException) as exc:
            raise LookupError(
                f"Cannot retrieve profile for @{username}: {exc}"
            ) from exc

        if response.data is None:
            raise LookupError(
                f"User @{username} does not exist or is suspended."
            )

        user_data = response.data
        metrics = user_data.public_metrics or {}

        return UserProfile(
            username=user_data.username,
            display_name=user_data.name or user_data.username,
            bio=user_data.description or "",
            followers_count=metrics.get("followers_count", 0),
            following_count=metrics.get("following_count", 0),
            tweet_count=metrics.get("tweet_count", 0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_user_id(self, username: str) -> str:
        """Resolve a username to a numeric user ID."""
        try:
            async with self._rate_limiter:
                response = self._client.get_user(
                    username=username,
                    user_fields=["id"],
                )
        except (BadRequest, Forbidden, TweepyException) as exc:
            raise LookupError(
                f"Cannot resolve user @{username}: {exc}"
            ) from exc

        if response.data is None:
            raise LookupError(
                f"User @{username} does not exist or is suspended."
            )

        return str(response.data.id)

    @staticmethod
    def _build_referenced_tweets_map(includes: dict[str, Any]) -> dict[str, Any]:
        """Build a map of referenced tweet ID -> tweet data from includes."""
        result: dict[str, Any] = {}
        for tweet in includes.get("tweets", []):
            result[str(tweet.id)] = tweet
        return result

    @staticmethod
    def _build_media_map(includes: dict[str, Any]) -> dict[str, str]:
        """Build a map of media_key -> media type from includes."""
        result: dict[str, str] = {}
        for media in includes.get("media", []):
            result[media.media_key] = media.type
        return result

    @staticmethod
    def _build_users_map(includes: dict[str, Any]) -> dict[str, str]:
        """Build a map of user ID -> username from includes."""
        result: dict[str, str] = {}
        for user in includes.get("users", []):
            result[str(user.id)] = user.username
        return result

    def _parse_tweet(
        self,
        tweet_data: Any,
        *,
        referenced_tweets_map: dict[str, Any],
        media_map: dict[str, str],
        users_map: dict[str, str],
    ) -> RawTweet:
        """Parse a tweepy tweet object into a RawTweet model."""
        # --- Basic fields ---
        tweet_id = str(tweet_data.id)
        text = tweet_data.text or ""
        created_at = tweet_data.created_at or datetime.now(timezone.utc)
        lang = tweet_data.lang

        # --- Engagement metrics ---
        metrics = tweet_data.public_metrics or {}
        like_count = metrics.get("like_count", 0)
        retweet_count = metrics.get("retweet_count", 0)
        reply_count = metrics.get("reply_count", 0)

        # --- Referenced tweets (reply/retweet/quote detection) ---
        is_reply = False
        is_retweet = False
        is_quote_tweet = False
        reply_to_user: str | None = None
        reply_to_tweet_id: str | None = None
        quoted_text: str | None = None

        referenced_tweets = tweet_data.referenced_tweets or []
        for ref in referenced_tweets:
            ref_type = ref.type if hasattr(ref, "type") else ref.get("type", "")
            ref_id = str(ref.id) if hasattr(ref, "id") else str(ref.get("id", ""))

            if ref_type == "replied_to":
                is_reply = True
                reply_to_tweet_id = ref_id
                # Try to get the reply-to user from in_reply_to_user_id
                in_reply_to_uid = getattr(tweet_data, "in_reply_to_user_id", None)
                if in_reply_to_uid:
                    reply_to_user = users_map.get(str(in_reply_to_uid))
            elif ref_type == "retweeted":
                is_retweet = True
            elif ref_type == "quoted":
                is_quote_tweet = True
                quoted_tweet = referenced_tweets_map.get(ref_id)
                if quoted_tweet is not None:
                    quoted_text = getattr(quoted_tweet, "text", None)

        # --- Entities parsing ---
        entities = tweet_data.entities or {}
        hashtags = self._parse_hashtags(entities)
        mentions = self._parse_mentions(entities)
        urls = self._parse_urls(entities)

        # --- Media types ---
        media_types: list[str] = []
        attachments = getattr(tweet_data, "attachments", None) or {}
        media_keys = attachments.get("media_keys", []) if isinstance(attachments, dict) else []
        for key in media_keys:
            media_type = media_map.get(key)
            if media_type:
                # Normalize: "animated_gif" -> "gif", "photo" -> "image"
                if media_type == "animated_gif":
                    media_types.append("gif")
                elif media_type == "photo":
                    media_types.append("image")
                else:
                    media_types.append(media_type)

        return RawTweet(
            id=tweet_id,
            text=text,
            created_at=created_at,
            is_reply=is_reply,
            reply_to_user=reply_to_user,
            reply_to_tweet_id=reply_to_tweet_id,
            is_retweet=is_retweet,
            is_quote_tweet=is_quote_tweet,
            quoted_text=quoted_text,
            hashtags=hashtags,
            mentions=mentions,
            urls=urls,
            media_types=media_types,
            like_count=like_count,
            retweet_count=retweet_count,
            reply_count=reply_count,
            language=lang,
        )

    @staticmethod
    def _parse_hashtags(entities: dict[str, Any]) -> list[str]:
        """Extract hashtag text values from the entities dict."""
        return [
            ht.get("tag", "") for ht in entities.get("hashtags", []) if ht.get("tag")
        ]

    @staticmethod
    def _parse_mentions(entities: dict[str, Any]) -> list[str]:
        """Extract mentioned usernames from the entities dict."""
        return [
            m.get("username", "") for m in entities.get("mentions", []) if m.get("username")
        ]

    @staticmethod
    def _parse_urls(entities: dict[str, Any]) -> list[str]:
        """Extract expanded URLs from the entities dict."""
        result: list[str] = []
        for url_obj in entities.get("urls", []):
            expanded = url_obj.get("expanded_url") or url_obj.get("url", "")
            if expanded:
                result.append(expanded)
        return result
