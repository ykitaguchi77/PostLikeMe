"""Async SQLite tweet cache for collected tweet data.

Provides persistent, query-efficient storage of :class:`RawTweet` objects so
that repeated analysis runs do not require re-fetching tweets from the API.
Uses ``aiosqlite`` for non-blocking database operations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

from postlikeme.models.raw_tweet import RawTweet

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tweets (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    collected_at TEXT NOT NULL
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_tweets_username
ON tweets (username, collected_at)
"""


class TweetCache:
    """Async SQLite cache for collected tweet data.

    Tweets are serialised as JSON and stored with a collection timestamp
    so that stale entries can be filtered out by ``max_age_days``.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  The file (and parent directories)
        will be created if they do not exist.

    Example
    -------
    >>> cache = TweetCache(Path("~/.postlikeme/cache/tweets.db"))
    >>> await cache.init_db()
    >>> await cache.store_tweets(raw_tweets, username="elonmusk")
    >>> cached = await cache.get_tweets("elonmusk", max_age_days=7)
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        """Create the tweets table and indices if they do not already exist."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(_CREATE_TABLE_SQL)
            await db.execute(_CREATE_INDEX_SQL)
            await db.commit()
        logger.debug("Tweet cache database initialised at %s.", self._db_path)

    async def store_tweets(self, tweets: list[RawTweet], username: str) -> None:
        """Store a batch of tweets in the cache.

        Existing tweets with the same ID are replaced (upserted) so that
        metric updates (likes, retweets) are reflected.

        Parameters
        ----------
        tweets:
            The raw tweets to store.
        username:
            The Twitter handle associated with these tweets (used for
            cache partitioning and lookup).
        """
        if not tweets:
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        username_lower = username.lower()

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO tweets (id, username, raw_json, collected_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        tweet.id,
                        username_lower,
                        tweet.model_dump_json(),
                        now_iso,
                    )
                    for tweet in tweets
                ],
            )
            await db.commit()

        logger.info(
            "Cached %d tweets for @%s in %s.",
            len(tweets),
            username,
            self._db_path.name,
        )

    async def get_tweets(
        self, username: str, max_age_days: int = 7
    ) -> list[RawTweet]:
        """Retrieve cached tweets for a user, filtered by freshness.

        Parameters
        ----------
        username:
            Twitter handle to look up (case-insensitive).
        max_age_days:
            Maximum age of cached entries in days.  Tweets collected more
            than this many days ago are excluded.  Use ``0`` or a negative
            value to return all cached tweets regardless of age.

        Returns
        -------
        list[RawTweet]
            Cached tweets sorted newest first (by ``created_at``).
        """
        username_lower = username.lower()

        async with aiosqlite.connect(str(self._db_path)) as db:
            if max_age_days > 0:
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(days=max_age_days)
                ).isoformat()
                cursor = await db.execute(
                    """
                    SELECT raw_json FROM tweets
                    WHERE username = ? AND collected_at >= ?
                    ORDER BY collected_at DESC
                    """,
                    (username_lower, cutoff),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT raw_json FROM tweets
                    WHERE username = ?
                    ORDER BY collected_at DESC
                    """,
                    (username_lower,),
                )

            rows = await cursor.fetchall()

        tweets: list[RawTweet] = []
        for (raw_json,) in rows:
            try:
                tweet = RawTweet.model_validate_json(raw_json)
                tweets.append(tweet)
            except Exception:
                logger.warning(
                    "Skipping corrupt cache entry for @%s.", username, exc_info=True
                )

        # Sort by created_at descending (newest first)
        tweets.sort(key=lambda t: t.created_at, reverse=True)

        logger.debug(
            "Retrieved %d cached tweets for @%s (max_age=%d days).",
            len(tweets),
            username,
            max_age_days,
        )
        return tweets

    async def get_tweet_count(self, username: str) -> int:
        """Return the number of cached tweets for a user.

        Parameters
        ----------
        username:
            Twitter handle to look up (case-insensitive).

        Returns
        -------
        int
            Number of tweets in the cache for this user.
        """
        username_lower = username.lower()

        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM tweets WHERE username = ?",
                (username_lower,),
            )
            row = await cursor.fetchone()

        return row[0] if row else 0

    async def clear_cache(self, username: str | None = None) -> None:
        """Delete cached tweets.

        Parameters
        ----------
        username:
            If provided, only delete tweets for this user.
            If ``None``, delete all cached tweets.
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            if username is not None:
                username_lower = username.lower()
                await db.execute(
                    "DELETE FROM tweets WHERE username = ?",
                    (username_lower,),
                )
                logger.info("Cleared tweet cache for @%s.", username)
            else:
                await db.execute("DELETE FROM tweets")
                logger.info("Cleared entire tweet cache.")
            await db.commit()
