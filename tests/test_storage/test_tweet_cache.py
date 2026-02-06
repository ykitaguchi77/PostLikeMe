"""Tests for the async SQLite tweet cache."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from postlikeme.models.raw_tweet import RawTweet
from postlikeme.storage.tweet_cache import TweetCache


def _make_tweet(tweet_id: str, text: str = "Test tweet") -> RawTweet:
    """Helper to create a minimal RawTweet for testing."""
    return RawTweet(
        id=tweet_id,
        text=text,
        created_at=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        language="en",
    )


class TestTweetCache:
    @pytest.fixture
    async def cache(self, tmp_path) -> TweetCache:
        c = TweetCache(tmp_path / "test_cache.db")
        await c.init_db()
        return c

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, cache):
        tweets = [_make_tweet("1", "Hello"), _make_tweet("2", "World")]
        await cache.store_tweets(tweets, username="testuser")

        cached = await cache.get_tweets("testuser", max_age_days=0)
        assert len(cached) == 2

        cached_texts = {t.text for t in cached}
        assert "Hello" in cached_texts
        assert "World" in cached_texts

    @pytest.mark.asyncio
    async def test_get_tweet_count(self, cache):
        tweets = [_make_tweet(str(i)) for i in range(5)]
        await cache.store_tweets(tweets, username="counter")

        count = await cache.get_tweet_count("counter")
        assert count == 5

    @pytest.mark.asyncio
    async def test_clear_cache_for_user(self, cache):
        tweets = [_make_tweet("1"), _make_tweet("2")]
        await cache.store_tweets(tweets, username="clearme")

        count_before = await cache.get_tweet_count("clearme")
        assert count_before == 2

        await cache.clear_cache(username="clearme")

        count_after = await cache.get_tweet_count("clearme")
        assert count_after == 0

    @pytest.mark.asyncio
    async def test_clear_all_cache(self, cache):
        await cache.store_tweets([_make_tweet("1")], username="user1")
        await cache.store_tweets([_make_tweet("2")], username="user2")

        await cache.clear_cache(username=None)

        assert await cache.get_tweet_count("user1") == 0
        assert await cache.get_tweet_count("user2") == 0

    @pytest.mark.asyncio
    async def test_upsert_replaces_existing(self, cache):
        tweet_v1 = _make_tweet("same_id", "Version 1")
        await cache.store_tweets([tweet_v1], username="upserter")

        tweet_v2 = _make_tweet("same_id", "Version 2")
        await cache.store_tweets([tweet_v2], username="upserter")

        count = await cache.get_tweet_count("upserter")
        assert count == 1

        cached = await cache.get_tweets("upserter", max_age_days=0)
        assert len(cached) == 1
        assert cached[0].text == "Version 2"

    @pytest.mark.asyncio
    async def test_empty_store_no_error(self, cache):
        await cache.store_tweets([], username="empty")
        count = await cache.get_tweet_count("empty")
        assert count == 0

    @pytest.mark.asyncio
    async def test_case_insensitive_username(self, cache):
        tweets = [_make_tweet("1", "Case test")]
        await cache.store_tweets(tweets, username="CamelCase")

        cached = await cache.get_tweets("camelcase", max_age_days=0)
        assert len(cached) == 1

        count = await cache.get_tweet_count("CAMELCASE")
        assert count == 1
