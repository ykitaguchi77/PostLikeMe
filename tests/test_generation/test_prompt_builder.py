"""Tests for prompt builder and Jinja2 template rendering."""

from __future__ import annotations

import pytest

from postlikeme.generation.prompt_builder import PromptBuilder


class TestPromptBuilder:
    """Verify Jinja2 templates render without errors and include profile data."""

    @pytest.fixture
    def builder(self):
        return PromptBuilder()

    def test_system_prompt_renders(self, builder, sample_voice_profile):
        prompt = builder.build_system_prompt(sample_voice_profile)
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial
        assert sample_voice_profile.username in prompt

    def test_tweet_prompt_with_topic(self, builder, sample_voice_profile):
        prompt = builder.build_tweet_prompt(sample_voice_profile, topic="AI safety")
        assert isinstance(prompt, str)
        assert "AI safety" in prompt

    def test_tweet_prompt_without_topic(self, builder, sample_voice_profile):
        prompt = builder.build_tweet_prompt(sample_voice_profile, topic=None)
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_reply_prompt_renders(self, builder, sample_voice_profile):
        target = "What do you think about the new Python release?"
        prompt = builder.build_reply_prompt(sample_voice_profile, target_tweet=target)
        assert isinstance(prompt, str)
        assert target in prompt

    def test_profile_data_in_system_prompt(self, builder, sample_voice_profile):
        prompt = builder.build_system_prompt(sample_voice_profile)
        # Username should appear in the prompt
        assert sample_voice_profile.username in prompt

    def test_example_tweets_in_prompt(self, builder, sample_voice_profile):
        prompt = builder.build_tweet_prompt(sample_voice_profile)
        # At least one example tweet should appear
        for example in sample_voice_profile.example_tweets.originals:
            if example in prompt:
                return  # Found at least one
        # If voice profile has examples, they should appear
        if sample_voice_profile.example_tweets.originals:
            assert any(ex in prompt for ex in sample_voice_profile.example_tweets.originals)
