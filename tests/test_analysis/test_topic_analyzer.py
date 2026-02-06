"""Tests for topic analysis helpers."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from postlikeme.analysis.topic_analyzer import (
    TopicAnalyzer,
    _recency_weight,
    _sentiment_label,
)
from postlikeme.utils.constants import DEFAULT_HALF_LIFE_DAYS


class TestRecencyWeight:
    def test_weight_at_zero_days(self):
        """Tweet from right now should have weight ~1.0."""
        now = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        weight = _recency_weight(now, now)
        assert weight == pytest.approx(1.0, abs=1e-6)

    def test_weight_at_half_life(self):
        """Tweet from exactly half_life_days ago should have weight ~0.5."""
        now = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)
        tweet_date = datetime(2024, 4, 16, 12, 0, tzinfo=timezone.utc)
        # ~90 days ago
        weight = _recency_weight(tweet_date, now, half_life_days=90)
        # Should be close to 0.5 (not exact due to month-day variations)
        assert 0.45 < weight < 0.55

    def test_weight_decreases_with_age(self):
        """Older tweets should have lower weights."""
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)
        recent = datetime(2024, 5, 25, tzinfo=timezone.utc)
        older = datetime(2024, 3, 1, tzinfo=timezone.utc)
        oldest = datetime(2023, 6, 1, tzinfo=timezone.utc)

        w_recent = _recency_weight(recent, now)
        w_older = _recency_weight(older, now)
        w_oldest = _recency_weight(oldest, now)

        assert w_recent > w_older > w_oldest
        assert w_recent <= 1.0
        assert w_oldest > 0.0

    def test_weight_formula(self):
        """Verify the exact formula: exp(-lambda * days_ago)."""
        now = datetime(2024, 1, 31, tzinfo=timezone.utc)
        tweet_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        days_ago = 30.0

        half_life = DEFAULT_HALF_LIFE_DAYS  # 90
        lambda_ = math.log(2) / half_life
        expected = math.exp(-lambda_ * days_ago)

        weight = _recency_weight(tweet_date, now, half_life_days=half_life)
        assert weight == pytest.approx(expected, rel=1e-4)

    def test_weight_always_positive(self):
        """Even for very old tweets, weight should be > 0."""
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        very_old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        weight = _recency_weight(very_old, now)
        assert weight > 0.0


class TestSentimentLabeling:
    def test_enthusiastic(self):
        assert _sentiment_label(0.5) == "enthusiastic"
        assert _sentiment_label(0.31) == "enthusiastic"

    def test_generally_positive(self):
        assert _sentiment_label(0.2) == "generally_positive"
        assert _sentiment_label(0.11) == "generally_positive"

    def test_neutral(self):
        assert _sentiment_label(0.0) == "neutral"
        assert _sentiment_label(0.05) == "neutral"
        assert _sentiment_label(-0.05) == "neutral"

    def test_critical(self):
        assert _sentiment_label(-0.2) == "critical"
        assert _sentiment_label(-0.15) == "critical"

    def test_negative(self):
        assert _sentiment_label(-0.5) == "negative"
        assert _sentiment_label(-0.31) == "negative"

    def test_boundary_values(self):
        # Exactly on boundaries
        assert _sentiment_label(0.3) == "generally_positive"  # not > 0.3
        assert _sentiment_label(0.1) == "neutral"  # not > 0.1
        assert _sentiment_label(-0.1) == "neutral"  # not > -0.1 is false, but > -0.1 is True
        assert _sentiment_label(-0.3) == "critical"  # not > -0.3


class TestTopicAnalyzerEmptyCorpus:
    def test_empty_corpus(self):
        """TopicAnalyzer should handle empty input gracefully."""
        analyzer = TopicAnalyzer()
        result = analyzer.analyze([])
        assert result.topics_ranked == []
        assert result.opinion_map == []
        assert result.semantic_clusters == []
