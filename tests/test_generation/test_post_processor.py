"""Tests for post-processing and style scoring of LLM output."""

from __future__ import annotations

import pytest

from postlikeme.generation.post_processor import PostProcessor


class TestPostProcessor:
    """Verify LLM output cleaning, validation, and scoring."""

    @pytest.fixture
    def processor(self):
        return PostProcessor()

    # --- Length validation ---

    def test_validate_length_short(self, processor):
        assert processor.validate_length("Hello world") is True

    def test_validate_length_exact_280(self, processor):
        text = "a" * 280
        assert processor.validate_length(text) is True

    def test_validate_length_over_280(self, processor):
        text = "a" * 281
        assert processor.validate_length(text) is False

    def test_validate_length_empty(self, processor):
        assert processor.validate_length("") is False

    def test_validate_length_whitespace_only(self, processor):
        assert processor.validate_length("   ") is False

    # --- LLM output cleaning ---

    def test_clean_strips_surrounding_quotes(self, processor):
        result = processor.clean_llm_output('"This is a tweet"')
        assert result == "This is a tweet"

    def test_clean_strips_smart_quotes(self, processor):
        result = processor.clean_llm_output("\u201cThis is a tweet\u201d")
        assert result == "This is a tweet"

    def test_clean_strips_label_prefix(self, processor):
        result = processor.clean_llm_output("Here's a tweet: This is great!")
        assert result == "This is great!"

    def test_clean_strips_tweet_prefix(self, processor):
        result = processor.clean_llm_output("Tweet: Hello world")
        assert result == "Hello world"

    def test_clean_handles_empty(self, processor):
        assert processor.clean_llm_output("") == ""

    def test_clean_preserves_normal_text(self, processor):
        text = "Just shipped a new feature! Really proud of the team"
        result = processor.clean_llm_output(text)
        assert result == text

    # --- Style adherence scoring ---

    def test_score_range(self, processor, sample_voice_profile):
        score = processor.score_style_adherence("Hello world!", sample_voice_profile)
        assert 0.0 <= score <= 1.0

    def test_score_empty_text(self, processor, sample_voice_profile):
        score = processor.score_style_adherence("", sample_voice_profile)
        assert score == 0.0

    # --- Candidate ranking ---

    def test_rank_candidates_sorted(self, processor, sample_voice_profile):
        candidates = [
            "a" * 300,  # Over 280 chars - should score low
            "Hello!",   # Short and simple
            "Just shipped a new feature! Really proud of the team",  # Matches style
        ]
        ranked = processor.rank_candidates(candidates, sample_voice_profile)
        assert len(ranked) >= 2  # At least 2 non-empty candidates
        # Verify sorted descending by score
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_candidates_returns_tuples(self, processor, sample_voice_profile):
        candidates = ["Hello world"]
        ranked = processor.rank_candidates(candidates, sample_voice_profile)
        assert len(ranked) == 1
        text, score = ranked[0]
        assert isinstance(text, str)
        assert isinstance(score, float)
