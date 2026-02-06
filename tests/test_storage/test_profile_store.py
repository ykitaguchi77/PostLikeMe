"""Tests for profile storage operations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from postlikeme.models.voice_profile import VoiceProfile
from postlikeme.storage.profile_store import ProfileStore


class TestProfileStore:
    @pytest.fixture
    def store(self, tmp_path) -> ProfileStore:
        return ProfileStore(tmp_path)

    @pytest.fixture
    def basic_profile(self) -> VoiceProfile:
        return VoiceProfile(
            username="testuser",
            display_name="Test User",
            bio="A test profile for storage tests",
            tweet_count_analyzed=100,
        )

    @pytest.fixture
    def second_profile(self) -> VoiceProfile:
        return VoiceProfile(
            username="anotheruser",
            display_name="Another User",
            bio="Second profile",
            tweet_count_analyzed=200,
        )

    def test_save_and_load_roundtrip(self, store, basic_profile):
        path = store.save_profile(basic_profile)
        assert path.exists()

        loaded = store.load_profile("testuser")
        assert loaded is not None
        assert loaded.username == basic_profile.username
        assert loaded.display_name == basic_profile.display_name
        assert loaded.bio == basic_profile.bio
        assert loaded.tweet_count_analyzed == basic_profile.tweet_count_analyzed

    def test_list_profiles(self, store, basic_profile, second_profile):
        store.save_profile(basic_profile)
        store.save_profile(second_profile)

        profiles = store.list_profiles()
        assert len(profiles) == 2
        assert "testuser" in profiles
        assert "anotheruser" in profiles

    def test_delete_profile(self, store, basic_profile):
        store.save_profile(basic_profile)
        assert store.load_profile("testuser") is not None

        result = store.delete_profile("testuser")
        assert result is True
        assert store.load_profile("testuser") is None

    def test_delete_nonexistent(self, store):
        result = store.delete_profile("nonexistent")
        assert result is False

    def test_load_nonexistent(self, store):
        loaded = store.load_profile("no_such_user")
        assert loaded is None

    def test_export_json(self, store, basic_profile):
        store.save_profile(basic_profile)
        json_str = store.export_profile("testuser", format="json")
        assert isinstance(json_str, str)

        data = json.loads(json_str)
        assert data["username"] == "testuser"
        assert data["display_name"] == "Test User"

    def test_export_markdown(self, store, basic_profile):
        store.save_profile(basic_profile)
        md = store.export_profile("testuser", format="markdown")
        assert isinstance(md, str)
        assert "# Voice Profile: @testuser" in md
        assert "## Metadata" in md
        assert "## Structural Style" in md
        assert "## Personality Profile" in md

    def test_export_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.export_profile("nonexistent")

    def test_export_invalid_format_raises(self, store, basic_profile):
        store.save_profile(basic_profile)
        with pytest.raises(ValueError, match="Unsupported export format"):
            store.export_profile("testuser", format="xml")

    def test_username_normalization(self, store, basic_profile):
        """Profile saved as 'testuser' should be loadable with '@TestUser'."""
        store.save_profile(basic_profile)
        loaded = store.load_profile("@TestUser")
        assert loaded is not None
        assert loaded.username == "testuser"

    def test_overwrite_existing(self, store):
        profile_v1 = VoiceProfile(
            username="overwrite_me",
            display_name="Version 1",
            tweet_count_analyzed=50,
        )
        store.save_profile(profile_v1)

        profile_v2 = VoiceProfile(
            username="overwrite_me",
            display_name="Version 2",
            tweet_count_analyzed=200,
        )
        store.save_profile(profile_v2)

        loaded = store.load_profile("overwrite_me")
        assert loaded is not None
        assert loaded.display_name == "Version 2"
        assert loaded.tweet_count_analyzed == 200
