"""Tests for data models."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from postlikeme.models.raw_tweet import CleanTweet, EmojiPosition, RawTweet, UserProfile
from postlikeme.models.voice_profile import (
    BigFiveScores,
    CapitalizationPatterns,
    EmojiProfile,
    ExampleTweets,
    HashtagProfile,
    OpinionEntry,
    PersonalityProfile,
    PunctuationPatterns,
    ReadabilityScores,
    ReplyStyle,
    SemanticCluster,
    StructuralStyle,
    TopicEntry,
    TopicProfile,
    TweetLengthDistribution,
    VoiceProfile,
)
from postlikeme.models.config import AppConfig


# ---------------------------------------------------------------------------
# RawTweet
# ---------------------------------------------------------------------------


class TestRawTweet:
    def test_creation_with_all_fields(self):
        tweet = RawTweet(
            id="1",
            text="Hello world",
            created_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            is_reply=False,
            reply_to_user=None,
            reply_to_tweet_id=None,
            is_retweet=False,
            is_quote_tweet=False,
            quoted_text=None,
            hashtags=["test"],
            mentions=["user1"],
            urls=["https://example.com"],
            media_types=["image"],
            like_count=42,
            retweet_count=8,
            reply_count=3,
            language="en",
        )
        assert tweet.id == "1"
        assert tweet.text == "Hello world"
        assert tweet.like_count == 42
        assert tweet.hashtags == ["test"]
        assert tweet.language == "en"

    def test_defaults(self):
        tweet = RawTweet(
            id="2",
            text="Minimal tweet",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert tweet.is_reply is False
        assert tweet.reply_to_user is None
        assert tweet.reply_to_tweet_id is None
        assert tweet.is_retweet is False
        assert tweet.is_quote_tweet is False
        assert tweet.quoted_text is None
        assert tweet.hashtags == []
        assert tweet.mentions == []
        assert tweet.urls == []
        assert tweet.media_types == []
        assert tweet.like_count == 0
        assert tweet.retweet_count == 0
        assert tweet.reply_count == 0
        assert tweet.language is None

    def test_json_serialization(self):
        tweet = RawTweet(
            id="3",
            text="Serialize me!",
            created_at=datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc),
            is_reply=True,
            reply_to_user="someone",
            reply_to_tweet_id="999",
            hashtags=["pydantic"],
            like_count=10,
            language="en",
        )
        json_str = tweet.model_dump_json()
        restored = RawTweet.model_validate_json(json_str)
        assert tweet == restored

    def test_frozen_model(self):
        tweet = RawTweet(
            id="4",
            text="Frozen",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        with pytest.raises(Exception):
            tweet.text = "Modified"

    def test_like_count_non_negative(self):
        with pytest.raises(Exception):
            RawTweet(
                id="5",
                text="Negative likes",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                like_count=-1,
            )


# ---------------------------------------------------------------------------
# CleanTweet
# ---------------------------------------------------------------------------


class TestCleanTweet:
    def test_creation(self):
        clean = CleanTweet(
            id="c1",
            text="Clean text here",
            original_text="Clean text here #tag",
            created_at=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            is_reply=False,
            hashtags=["tag"],
            like_count=5,
            language="en",
        )
        assert clean.id == "c1"
        assert clean.text == "Clean text here"
        assert clean.original_text == "Clean text here #tag"
        assert clean.hashtags == ["tag"]

    def test_emoji_positions(self):
        emoji_pos = EmojiPosition(
            emoji="\U0001f680",
            match_start=5,
            match_end=6,
            position_ratio=0.5,
        )
        clean = CleanTweet(
            id="c2",
            text="Hello world",
            original_text="Hello \U0001f680 world",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            emoji_positions=[emoji_pos],
        )
        assert len(clean.emoji_positions) == 1
        assert clean.emoji_positions[0].emoji == "\U0001f680"
        assert clean.emoji_positions[0].position_ratio == 0.5

    def test_reply_fields(self):
        clean = CleanTweet(
            id="c3",
            text="@USER Great point!",
            original_text="@johndoe Great point!",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            is_reply=True,
            reply_to_user="johndoe",
            reply_to_tweet_id="100",
            original_mentions=["johndoe"],
        )
        assert clean.is_reply is True
        assert clean.reply_to_user == "johndoe"
        assert clean.original_mentions == ["johndoe"]

    def test_json_roundtrip(self):
        clean = CleanTweet(
            id="c4",
            text="Roundtrip test",
            original_text="Roundtrip test",
            created_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            like_count=100,
        )
        json_str = clean.model_dump_json()
        restored = CleanTweet.model_validate_json(json_str)
        assert clean == restored


# ---------------------------------------------------------------------------
# EmojiPosition
# ---------------------------------------------------------------------------


class TestEmojiPosition:
    def test_position_ratio_bounds(self):
        ep = EmojiPosition(emoji="\U0001f525", match_start=0, match_end=1, position_ratio=0.0)
        assert ep.position_ratio == 0.0

        ep2 = EmojiPosition(emoji="\U0001f525", match_start=10, match_end=11, position_ratio=1.0)
        assert ep2.position_ratio == 1.0

    def test_position_ratio_out_of_range(self):
        with pytest.raises(Exception):
            EmojiPosition(emoji="\U0001f525", match_start=0, match_end=1, position_ratio=1.5)

    def test_negative_match_start(self):
        with pytest.raises(Exception):
            EmojiPosition(emoji="\U0001f525", match_start=-1, match_end=1, position_ratio=0.0)


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------


class TestUserProfile:
    def test_creation(self):
        profile = UserProfile(
            username="testuser",
            display_name="Test User",
            bio="A test profile",
            followers_count=1000,
            following_count=500,
            tweet_count=200,
        )
        assert profile.username == "testuser"
        assert profile.followers_count == 1000

    def test_defaults(self):
        profile = UserProfile(username="min", display_name="Min")
        assert profile.bio == ""
        assert profile.followers_count == 0
        assert profile.following_count == 0
        assert profile.tweet_count == 0


# ---------------------------------------------------------------------------
# VoiceProfile
# ---------------------------------------------------------------------------


class TestVoiceProfile:
    def test_json_roundtrip(self, sample_voice_profile):
        json_str = sample_voice_profile.to_json()
        restored = VoiceProfile.from_json(json_str)
        assert restored.username == sample_voice_profile.username
        assert restored.display_name == sample_voice_profile.display_name
        assert restored.tweet_count_analyzed == sample_voice_profile.tweet_count_analyzed

    def test_schema_version(self, sample_voice_profile):
        assert sample_voice_profile.schema_version == "1.0.0"

    def test_defaults(self):
        profile = VoiceProfile(username="default_user")
        assert profile.display_name == ""
        assert profile.bio == ""
        assert profile.schema_version == "1.0.0"
        assert profile.tweet_count_analyzed == 0
        assert isinstance(profile.structural_style, StructuralStyle)
        assert isinstance(profile.emoji_profile, EmojiProfile)
        assert isinstance(profile.personality_profile, PersonalityProfile)

    def test_to_json_file(self, sample_voice_profile, tmp_path):
        file_path = tmp_path / "test_profile.json"
        sample_voice_profile.to_json_file(file_path)
        assert file_path.exists()
        restored = VoiceProfile.from_json_file(file_path)
        assert restored.username == sample_voice_profile.username

    def test_structural_style_fields(self, sample_voice_profile):
        ss = sample_voice_profile.structural_style
        assert ss.avg_tweet_length == 142.5
        assert ss.question_ratio == 0.18
        assert ss.exclamation_ratio == 0.15
        assert ss.tweet_length_distribution is not None
        assert ss.tweet_length_distribution.mean == 142.5

    def test_personality_profile_fields(self, sample_voice_profile):
        pp = sample_voice_profile.personality_profile
        assert pp.formality_score == 0.35
        assert pp.humor_style == "dry"
        assert pp.tone == "casual"
        assert 0.0 <= pp.big_five.openness <= 1.0
        assert 0.0 <= pp.big_five.neuroticism <= 1.0


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------


class TestAppConfig:
    def test_defaults(self):
        config = AppConfig()
        assert config.default_llm_provider == "anthropic"
        assert config.default_llm_model == "claude-sonnet-4-20250514"
        assert config.default_tweet_count == 500
        assert config.default_temperature == 0.85
        assert config.default_collector == "api"

    def test_data_dir_creation(self, tmp_path):
        config = AppConfig(data_dir=tmp_path / "test_postlikeme")
        config.ensure_dirs()
        assert config.profiles_dir.exists()
        assert config.cache_dir.exists()
        assert config.logs_dir.exists()

    def test_derived_paths(self, tmp_path):
        config = AppConfig(data_dir=tmp_path / "data")
        assert config.profiles_dir == tmp_path / "data" / "profiles"
        assert config.cache_dir == tmp_path / "data" / "cache"
        assert config.logs_dir == tmp_path / "data" / "logs"
        assert config.config_path == tmp_path / "data" / "config.toml"

    def test_env_var_resolution(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
        config = AppConfig()
        assert config.anthropic_api_key == "test-key-123"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        config = AppConfig(anthropic_api_key="explicit-key")
        assert config.anthropic_api_key == "explicit-key"

    def test_temperature_bounds(self):
        with pytest.raises(Exception):
            AppConfig(default_temperature=3.0)
        with pytest.raises(Exception):
            AppConfig(default_temperature=-0.1)

    def test_tweet_count_minimum(self):
        with pytest.raises(Exception):
            AppConfig(default_tweet_count=0)
