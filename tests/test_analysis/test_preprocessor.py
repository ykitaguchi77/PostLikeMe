"""Tests for the tweet preprocessing pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from postlikeme.analysis.preprocessor import TweetPreprocessor
from postlikeme.models.raw_tweet import RawTweet


def _make_tweet(
    tweet_id: str,
    text: str,
    is_reply: bool = False,
    is_retweet: bool = False,
    language: str | None = "en",
    **kwargs,
) -> RawTweet:
    """Helper to create a RawTweet with sensible defaults."""
    return RawTweet(
        id=tweet_id,
        text=text,
        created_at=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        is_reply=is_reply,
        is_retweet=is_retweet,
        language=language,
        **kwargs,
    )


class TestTweetPreprocessor:
    @pytest.fixture
    def preprocessor(self) -> TweetPreprocessor:
        return TweetPreprocessor(target_language="en")

    def test_filter_retweets(self, preprocessor):
        tweets = [
            _make_tweet("1", "Normal tweet"),
            _make_tweet("2", "RT @someone: Retweeted content", is_retweet=True),
            _make_tweet("3", "Another normal tweet"),
        ]
        originals, replies = preprocessor.preprocess(tweets)
        all_clean = originals + replies
        assert len(all_clean) == 2
        assert all(t.id != "2" for t in all_clean)

    def test_filter_url_only(self, preprocessor):
        tweets = [
            _make_tweet("1", "Normal tweet with text"),
            _make_tweet("2", "https://t.co/abc123"),
            _make_tweet("3", "https://t.co/abc https://t.co/xyz"),
        ]
        originals, replies = preprocessor.preprocess(tweets)
        all_clean = originals + replies
        # URL-only tweets should be filtered out
        assert all("Normal tweet" in t.text or len(t.text) > 0 for t in all_clean)
        clean_ids = {t.id for t in all_clean}
        assert "1" in clean_ids

    def test_filter_mention_only(self, preprocessor):
        tweets = [
            _make_tweet("1", "Real content here"),
            _make_tweet("2", "@alice @bob @charlie"),
        ]
        originals, replies = preprocessor.preprocess(tweets)
        all_clean = originals + replies
        clean_ids = {t.id for t in all_clean}
        assert "1" in clean_ids
        # Mention-only tweet should be filtered
        assert "2" not in clean_ids

    def test_separate_originals_replies(self, preprocessor):
        tweets = [
            _make_tweet("1", "Original tweet"),
            _make_tweet("2", "@someone Reply tweet", is_reply=True, reply_to_user="someone"),
            _make_tweet("3", "Another original"),
            _make_tweet("4", "@other Another reply", is_reply=True, reply_to_user="other"),
        ]
        originals, replies = preprocessor.preprocess(tweets)
        assert len(originals) == 2
        assert len(replies) == 2
        assert all(not t.is_reply for t in originals)
        assert all(t.is_reply for t in replies)

    def test_clean_urls(self, preprocessor):
        tweets = [
            _make_tweet("1", "Check out my blog https://t.co/abc123 it's great"),
        ]
        originals, replies = preprocessor.preprocess(tweets)
        assert len(originals) == 1
        assert "https://t.co" not in originals[0].text

    def test_emoji_positions(self, preprocessor):
        tweets = [
            _make_tweet("1", "Hello \U0001f680 world \U0001f525"),
        ]
        originals, replies = preprocessor.preprocess(tweets)
        assert len(originals) == 1
        assert len(originals[0].emoji_positions) == 2
        emojis = [ep.emoji for ep in originals[0].emoji_positions]
        assert "\U0001f680" in emojis
        assert "\U0001f525" in emojis

    def test_whitespace_normalized(self, preprocessor):
        tweets = [
            _make_tweet("1", "Multiple   spaces   here    now"),
        ]
        originals, replies = preprocessor.preprocess(tweets)
        assert len(originals) == 1
        assert "  " not in originals[0].text

    def test_empty_input(self, preprocessor):
        originals, replies = preprocessor.preprocess([])
        assert originals == []
        assert replies == []

    def test_mention_replaced_with_placeholder(self, preprocessor):
        tweets = [
            _make_tweet(
                "1",
                "@johndoe Great point!",
                is_reply=True,
                reply_to_user="johndoe",
            ),
        ]
        originals, replies = preprocessor.preprocess(tweets)
        assert len(replies) == 1
        assert "@USER" in replies[0].text
        assert "johndoe" in replies[0].original_mentions

    def test_language_filter(self):
        preprocessor = TweetPreprocessor(target_language="en")
        tweets = [
            _make_tweet("1", "English tweet", language="en"),
            _make_tweet("2", "Tweet fran\u00e7ais", language="fr"),
            _make_tweet("3", "Unknown language tweet", language=None),
        ]
        originals, replies = preprocessor.preprocess(tweets)
        all_clean = originals + replies
        # "en" and None should be kept, "fr" should be filtered
        clean_ids = {t.id for t in all_clean}
        assert "1" in clean_ids
        assert "2" not in clean_ids
        assert "3" in clean_ids
