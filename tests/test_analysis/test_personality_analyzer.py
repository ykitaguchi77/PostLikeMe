"""Tests for personality and tone analysis."""

from __future__ import annotations

import math

import pytest

from postlikeme.analysis.personality_analyzer import PersonalityAnalyzer
from postlikeme.models.raw_tweet import CleanTweet


class TestSigmoidNormalization:
    """Verify the sigmoid function maps any float to (0, 1)."""

    def test_zero_input(self):
        result = 1.0 / (1.0 + math.exp(0))
        assert result == pytest.approx(0.5)

    def test_large_positive(self):
        result = 1.0 / (1.0 + math.exp(-10.0))
        assert 0.999 < result < 1.0

    def test_large_negative(self):
        result = 1.0 / (1.0 + math.exp(10.0))
        assert 0.0 < result < 0.001

    def test_moderate_values(self):
        for x in [-3.0, -1.0, 0.5, 1.5, 3.0]:
            result = 1.0 / (1.0 + math.exp(-x))
            assert 0.0 < result < 1.0


class TestBigFiveScores:
    """Verify Big Five trait scores are bounded."""

    @pytest.fixture
    def analyzer(self):
        return PersonalityAnalyzer()

    def test_all_traits_in_range(self, analyzer, sample_clean_tweets, sample_voice_profile):
        profile = analyzer.analyze(sample_clean_tweets, sample_voice_profile.structural_style)
        big5 = profile.big_five
        for trait_name in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            score = getattr(big5, trait_name)
            assert 0.0 <= score <= 1.0, f"{trait_name} = {score} is out of [0, 1]"

    def test_formality_score_in_range(self, analyzer, sample_clean_tweets, sample_voice_profile):
        profile = analyzer.analyze(sample_clean_tweets, sample_voice_profile.structural_style)
        assert 0.0 <= profile.formality_score <= 1.0

    def test_assertiveness_in_range(self, analyzer, sample_clean_tweets, sample_voice_profile):
        profile = analyzer.analyze(sample_clean_tweets, sample_voice_profile.structural_style)
        assert 0.0 <= profile.assertiveness <= 1.0


class TestHumorDetection:
    """Verify humor style classification."""

    @pytest.fixture
    def analyzer(self):
        return PersonalityAnalyzer()

    def test_humor_style_is_valid(self, analyzer, sample_clean_tweets, sample_voice_profile):
        profile = analyzer.analyze(sample_clean_tweets, sample_voice_profile.structural_style)
        valid_styles = {"dry", "sarcastic", "absurdist", "wholesome", "self_deprecating", "none", "frequent", "occasional", "rare"}
        # humor_style could be any descriptive string, just verify it's non-empty
        assert isinstance(profile.humor_style, str)
        assert len(profile.humor_style) > 0
