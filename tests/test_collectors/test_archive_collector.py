"""Tests for the archive-based tweet collector."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from postlikeme.collectors.archive_collector import ArchiveCollector
from postlikeme.models.raw_tweet import RawTweet


class TestArchiveCollectorCSV:
    @pytest.fixture
    def csv_file(self, tmp_path) -> Path:
        """Create a temporary CSV file with sample tweets."""
        file_path = tmp_path / "tweets.csv"
        rows = [
            {
                "id": "1",
                "text": "Hello from CSV!",
                "created_at": "2024-01-15T10:00:00+00:00",
                "is_reply": "false",
                "reply_to_user": "",
                "like_count": "10",
                "retweet_count": "2",
                "reply_count": "1",
            },
            {
                "id": "2",
                "text": "@someone Great point!",
                "created_at": "2024-01-14T09:00:00+00:00",
                "is_reply": "true",
                "reply_to_user": "someone",
                "like_count": "5",
                "retweet_count": "0",
                "reply_count": "0",
            },
            {
                "id": "3",
                "text": "Another original tweet here",
                "created_at": "2024-01-13T08:00:00+00:00",
                "is_reply": "false",
                "reply_to_user": "",
                "like_count": "20",
                "retweet_count": "5",
                "reply_count": "3",
            },
            {
                "id": "4",
                "text": "RT @other: A retweet",
                "created_at": "2024-01-12T07:00:00+00:00",
                "is_reply": "false",
                "reply_to_user": "",
                "like_count": "0",
                "retweet_count": "0",
                "reply_count": "0",
            },
        ]
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return file_path

    @pytest.mark.asyncio
    async def test_load_csv(self, csv_file):
        collector = ArchiveCollector(csv_file)
        tweets = await collector.collect_user_tweets("testuser", count=100)
        assert len(tweets) == 4
        assert all(isinstance(t, RawTweet) for t in tweets)

    @pytest.mark.asyncio
    async def test_collect_with_count_limit(self, csv_file):
        collector = ArchiveCollector(csv_file)
        tweets = await collector.collect_user_tweets("testuser", count=2)
        assert len(tweets) == 2

    @pytest.mark.asyncio
    async def test_exclude_replies(self, csv_file):
        collector = ArchiveCollector(csv_file)
        tweets = await collector.collect_user_tweets(
            "testuser", count=100, include_replies=False
        )
        assert all(not t.is_reply for t in tweets)

    @pytest.mark.asyncio
    async def test_retweet_detection_csv(self, csv_file):
        collector = ArchiveCollector(csv_file)
        tweets = await collector.collect_user_tweets("testuser", count=100)
        rt_tweets = [t for t in tweets if t.is_retweet]
        assert len(rt_tweets) == 1
        assert rt_tweets[0].text.startswith("RT @")


class TestArchiveCollectorJSON:
    @pytest.fixture
    def json_file(self, tmp_path) -> Path:
        """Create a temporary JSON file in Twitter export format."""
        file_path = tmp_path / "tweets.json"
        data = [
            {
                "tweet": {
                    "id_str": "1001",
                    "full_text": "A tweet from the archive",
                    "created_at": "Mon Jan 15 10:30:00 +0000 2024",
                    "favorite_count": 42,
                    "retweet_count": 8,
                    "lang": "en",
                    "entities": {
                        "hashtags": [{"text": "test"}],
                        "user_mentions": [],
                        "urls": [],
                    },
                }
            },
            {
                "tweet": {
                    "id_str": "1002",
                    "full_text": "@user Reply tweet here",
                    "created_at": "Sun Jan 14 09:00:00 +0000 2024",
                    "in_reply_to_screen_name": "user",
                    "in_reply_to_status_id_str": "999",
                    "favorite_count": 5,
                    "retweet_count": 0,
                    "lang": "en",
                    "entities": {
                        "hashtags": [],
                        "user_mentions": [{"screen_name": "user"}],
                        "urls": [],
                    },
                }
            },
            {
                "tweet": {
                    "id_str": "1003",
                    "full_text": "Another standalone tweet",
                    "created_at": "Sat Jan 13 08:00:00 +0000 2024",
                    "favorite_count": 100,
                    "retweet_count": 20,
                    "lang": "en",
                    "entities": {
                        "hashtags": [],
                        "user_mentions": [],
                        "urls": [],
                    },
                }
            },
        ]
        file_path.write_text(json.dumps(data), encoding="utf-8")
        return file_path

    @pytest.mark.asyncio
    async def test_load_json(self, json_file):
        collector = ArchiveCollector(json_file)
        tweets = await collector.collect_user_tweets("testuser", count=100)
        assert len(tweets) == 3
        assert all(isinstance(t, RawTweet) for t in tweets)

    @pytest.mark.asyncio
    async def test_json_hashtags_parsed(self, json_file):
        collector = ArchiveCollector(json_file)
        tweets = await collector.collect_user_tweets("testuser", count=100)
        tweet_with_tag = [t for t in tweets if t.hashtags]
        assert len(tweet_with_tag) == 1
        assert "test" in tweet_with_tag[0].hashtags

    @pytest.mark.asyncio
    async def test_json_reply_detection(self, json_file):
        collector = ArchiveCollector(json_file)
        tweets = await collector.collect_user_tweets("testuser", count=100)
        replies = [t for t in tweets if t.is_reply]
        assert len(replies) == 1
        assert replies[0].reply_to_user == "user"

    @pytest.mark.asyncio
    async def test_get_user_profile(self, json_file):
        collector = ArchiveCollector(json_file)
        profile = await collector.get_user_profile("testuser")
        assert profile.username == "testuser"
        assert profile.tweet_count == 3


class TestArchiveCollectorEdgeCases:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ArchiveCollector(tmp_path / "nonexistent.json")

    def test_unsupported_format(self, tmp_path):
        txt_file = tmp_path / "tweets.txt"
        txt_file.write_text("some data")
        with pytest.raises(ValueError, match="Unsupported archive format"):
            ArchiveCollector(txt_file)

    @pytest.mark.asyncio
    async def test_empty_json_array(self, tmp_path):
        file_path = tmp_path / "empty.json"
        file_path.write_text("[]", encoding="utf-8")
        collector = ArchiveCollector(file_path)
        tweets = await collector.collect_user_tweets("testuser", count=100)
        assert tweets == []

    @pytest.mark.asyncio
    async def test_collect_tweet_by_id(self, tmp_path):
        file_path = tmp_path / "byid.json"
        data = [
            {
                "id_str": "42",
                "full_text": "Found me!",
                "created_at": "2024-01-15T10:00:00+00:00",
                "lang": "en",
                "entities": {"hashtags": [], "user_mentions": [], "urls": []},
            }
        ]
        file_path.write_text(json.dumps(data), encoding="utf-8")
        collector = ArchiveCollector(file_path)
        tweet = await collector.collect_tweet_by_id("42")
        assert tweet.id == "42"
        assert tweet.text == "Found me!"

    @pytest.mark.asyncio
    async def test_collect_tweet_by_id_not_found(self, tmp_path):
        file_path = tmp_path / "missing.json"
        file_path.write_text("[]", encoding="utf-8")
        collector = ArchiveCollector(file_path)
        with pytest.raises(LookupError):
            await collector.collect_tweet_by_id("999")
