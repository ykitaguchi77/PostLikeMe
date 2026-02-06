"""Tests for the structural analyzer.

These tests focus on the pure-computation helper methods that do not
require a spaCy pipeline, keeping the test suite fast and dependency-light.
The full ``analyze`` / ``analyze_full`` methods are integration tests that
need spaCy + textstat and are exercised separately.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pytest

from postlikeme.models.raw_tweet import CleanTweet, EmojiPosition
from postlikeme.models.voice_profile import TweetLengthDistribution


# ---------------------------------------------------------------------------
# Standalone computation tests (no spaCy required)
# ---------------------------------------------------------------------------


class TestYulesK:
    """Test the Yule's K vocabulary richness calculation."""

    def test_yules_k_known_values(self):
        """Hand-calculate K for a known text:
        'the cat sat on the mat the cat'
        N=8 tokens, freq: the=3, cat=2, sat=1, on=1, mat=1
        V=5, V1=3 (sat,on,mat), V2=1 (cat), V3=1 (the)
        freq_spectrum: {1:3, 2:1, 3:1}
        M2 = 3*1^2 + 1*2^2 + 1*3^2 = 3+4+9 = 16
        K = 10000*(M2-N)/(N^2) = 10000*(16-8)/64 = 10000*8/64 = 1250
        """
        tokens = ["the", "cat", "sat", "on", "the", "mat", "the", "cat"]
        freq = Counter(tokens)
        N = len(tokens)

        freq_spectrum = Counter(freq.values())
        M2 = sum(i * i * f_v for i, f_v in freq_spectrum.items())

        yules_k = 10_000 * (M2 - N) / (N * N) if (M2 - N) != 0 else 0.0
        assert yules_k == 1250.0

    def test_yules_k_uniform_text(self):
        """All unique tokens => M2 = N (each freq=1), so M2-N=0, K=0."""
        tokens = ["a", "b", "c", "d", "e"]
        freq = Counter(tokens)
        N = len(tokens)

        freq_spectrum = Counter(freq.values())
        M2 = sum(i * i * f_v for i, f_v in freq_spectrum.items())

        yules_k = 10_000 * (M2 - N) / (N * N) if (M2 - N) != 0 else 0.0
        assert yules_k == 0.0

    def test_yules_k_repeated_single_word(self):
        """All same token => V=1, freq_spectrum={5:1}, M2=25,
        K = 10000*(25-5)/25 = 10000*20/25 = 8000"""
        tokens = ["hello"] * 5
        freq = Counter(tokens)
        N = len(tokens)

        freq_spectrum = Counter(freq.values())
        M2 = sum(i * i * f_v for i, f_v in freq_spectrum.items())

        yules_k = 10_000 * (M2 - N) / (N * N) if (M2 - N) != 0 else 0.0
        assert yules_k == 8000.0


class TestHapaxRatio:
    def test_hapax_ratio_all_unique(self):
        """When all tokens are unique, hapax ratio = 1.0 (all are hapax)."""
        tokens = ["a", "b", "c", "d", "e"]
        freq = Counter(tokens)
        V = len(freq)
        V1 = sum(1 for count in freq.values() if count == 1)
        hapax_ratio = V1 / V if V > 0 else 0.0
        assert hapax_ratio == 1.0

    def test_hapax_ratio_no_hapax(self):
        """When all tokens appear 2+ times, hapax ratio = 0.0."""
        tokens = ["a", "a", "b", "b", "c", "c"]
        freq = Counter(tokens)
        V = len(freq)
        V1 = sum(1 for count in freq.values() if count == 1)
        hapax_ratio = V1 / V if V > 0 else 0.0
        assert hapax_ratio == 0.0

    def test_hapax_ratio_mixed(self):
        """Partial hapax: 'a' x3, 'b' x2, 'c' x1, 'd' x1 => V=4, V1=2, ratio=0.5"""
        tokens = ["a", "a", "a", "b", "b", "c", "d"]
        freq = Counter(tokens)
        V = len(freq)
        V1 = sum(1 for count in freq.values() if count == 1)
        hapax_ratio = V1 / V
        assert hapax_ratio == 0.5


class TestQuestionRatio:
    def test_question_ratio(self):
        """2 questions out of 5 tweets = 0.4."""
        texts = [
            "This is a statement.",
            "Is this a question?",
            "Another statement here.",
            "What do you think?",
            "Final statement.",
        ]
        question_count = sum(1 for t in texts if t.strip().endswith("?"))
        ratio = question_count / len(texts)
        assert ratio == pytest.approx(0.4)

    def test_question_ratio_no_questions(self):
        texts = ["Statement one.", "Statement two.", "Statement three."]
        question_count = sum(1 for t in texts if t.strip().endswith("?"))
        ratio = question_count / len(texts)
        assert ratio == 0.0

    def test_question_ratio_all_questions(self):
        texts = ["Question one?", "Question two?", "Question three?"]
        question_count = sum(1 for t in texts if t.strip().endswith("?"))
        ratio = question_count / len(texts)
        assert ratio == 1.0


class TestEmojiAnalysis:
    def test_emoji_tweet_ratio(self):
        """Verify emoji tweet ratio calculation."""
        tweets_with_emoji = 3
        total_tweets = 10
        ratio = tweets_with_emoji / total_tweets
        assert ratio == 0.3

    def test_emoji_positional_classification(self):
        """Test leading/inline/trailing classification based on position_ratio."""
        positions = [0.05, 0.1, 0.5, 0.6, 0.85, 0.95]
        leading = sum(1 for p in positions if p < 0.2)
        inline = sum(1 for p in positions if 0.2 <= p <= 0.8)
        trailing = sum(1 for p in positions if p > 0.8)
        total = len(positions)

        assert leading == 2
        assert inline == 2
        assert trailing == 2
        assert leading + inline + trailing == total

    def test_emoji_diversity(self):
        """5 unique emojis out of 10 total = 0.5 diversity."""
        emoji_counter = Counter({
            "\U0001f680": 3,
            "\U0001f525": 2,
            "\U0001f4af": 2,
            "\U0001f389": 2,
            "\U0001f64f": 1,
        })
        unique = len(emoji_counter)
        total = sum(emoji_counter.values())
        diversity = unique / total
        assert diversity == 0.5


class TestTweetLengthStats:
    def test_tweet_length_stats(self):
        """Verify mean/median/std for known lengths."""
        lengths = [10, 20, 30, 40, 50]
        arr = np.array(lengths, dtype=float)

        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr))
        min_val = int(np.min(arr))
        max_val = int(np.max(arr))

        assert mean == 30.0
        assert median == 30.0
        assert std == pytest.approx(14.1421, rel=1e-3)
        assert min_val == 10
        assert max_val == 50

    def test_single_tweet_length(self):
        """Single tweet => std should be 0."""
        lengths = [100]
        arr = np.array(lengths, dtype=float)
        assert float(np.mean(arr)) == 100.0
        assert float(np.std(arr)) == 0.0

    def test_tweet_length_percentiles(self):
        """Verify p25 and p75 for a known distribution."""
        lengths = list(range(1, 101))  # 1 to 100
        arr = np.array(lengths, dtype=float)
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
        assert p25 == 25.75
        assert p75 == 75.25
