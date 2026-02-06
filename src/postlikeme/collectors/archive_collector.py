"""Archive-based collector for offline Twitter data imports.

Implements :class:`BaseCollector` for local data files, supporting two formats:

1. **Twitter official data export** -- the JSON format you get from
   "Settings > Your Account > Download an archive of your data".  This is a
   JSON array where each item has a ``"tweet"`` object with ``"full_text"``,
   ``"created_at"``, ``"id_str"``, ``"entities"``, etc.

2. **Simple CSV** -- a flat CSV with columns:
   ``id, text, created_at, is_reply, reply_to_user, like_count, retweet_count, reply_count``
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from postlikeme.collectors.base import BaseCollector
from postlikeme.models.raw_tweet import RawTweet, UserProfile

logger = logging.getLogger(__name__)

# Twitter's archive format uses this strftime pattern for created_at
_TWITTER_ARCHIVE_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"

# Common alternative date formats to try when parsing
_FALLBACK_DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d",
]


class ArchiveCollector(BaseCollector):
    """Collect tweets from a local Twitter data export or CSV file.

    Parameters
    ----------
    file_path:
        Path to the archive file.  Must be a ``.json`` or ``.csv`` file.
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise FileNotFoundError(f"Archive file not found: {self._file_path}")

        suffix = self._file_path.suffix.lower()
        if suffix not in (".json", ".csv"):
            raise ValueError(
                f"Unsupported archive format '{suffix}'. "
                "Expected .json (Twitter export) or .csv."
            )

        self._tweets: list[RawTweet] | None = None  # Lazy-loaded cache

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    async def collect_user_tweets(
        self,
        username: str,
        count: int = 500,
        include_replies: bool = True,
    ) -> list[RawTweet]:
        """Read tweets from the archive file.

        The *username* parameter is accepted for interface compatibility but
        is not used for filtering (the archive is assumed to belong to a single
        user).  Set ``include_replies=False`` to exclude reply tweets.

        Parameters
        ----------
        username:
            Ignored for archive imports (kept for interface compatibility).
        count:
            Maximum number of tweets to return.
        include_replies:
            Whether to include replies in the result.

        Returns
        -------
        list[RawTweet]
            Tweets sorted in reverse-chronological order.
        """
        all_tweets = await self._load_tweets()

        if not include_replies:
            all_tweets = [t for t in all_tweets if not t.is_reply]

        # Sort newest first
        sorted_tweets = sorted(all_tweets, key=lambda t: t.created_at, reverse=True)

        result = sorted_tweets[:count]
        logger.info(
            "Loaded %d tweets from archive %s (requested %d).",
            len(result),
            self._file_path.name,
            count,
        )
        return result

    async def collect_tweet_by_id(self, tweet_id: str) -> RawTweet:
        """Search the loaded archive for a tweet with the given ID.

        Parameters
        ----------
        tweet_id:
            The tweet identifier to search for.

        Returns
        -------
        RawTweet

        Raises
        ------
        LookupError
            If no tweet with the given ID exists in the archive.
        """
        all_tweets = await self._load_tweets()
        for tweet in all_tweets:
            if tweet.id == tweet_id:
                return tweet
        raise LookupError(
            f"Tweet {tweet_id} not found in archive {self._file_path.name}."
        )

    async def get_user_profile(self, username: str) -> UserProfile:
        """Return a placeholder user profile derived from file metadata.

        Since archive files do not contain full profile information, this
        returns a minimal profile with the tweet count inferred from the data.

        Parameters
        ----------
        username:
            The username to assign to the profile.

        Returns
        -------
        UserProfile
        """
        all_tweets = await self._load_tweets()
        return UserProfile(
            username=username,
            display_name=username,
            bio=f"Imported from archive: {self._file_path.name}",
            followers_count=0,
            following_count=0,
            tweet_count=len(all_tweets),
        )

    # ------------------------------------------------------------------
    # Internal: loading & parsing
    # ------------------------------------------------------------------

    async def _load_tweets(self) -> list[RawTweet]:
        """Load and cache tweets from the file (lazy, called once)."""
        if self._tweets is not None:
            return self._tweets

        suffix = self._file_path.suffix.lower()
        if suffix == ".json":
            self._tweets = self._load_json()
        else:
            self._tweets = self._load_csv()

        logger.info(
            "Parsed %d tweets from %s.", len(self._tweets), self._file_path.name
        )
        return self._tweets

    # --- JSON (Twitter official export) ---

    def _load_json(self) -> list[RawTweet]:
        """Parse the Twitter official data export JSON format."""
        with open(self._file_path, "r", encoding="utf-8") as fh:
            raw_data = json.load(fh)

        if not isinstance(raw_data, list):
            raise ValueError(
                f"Expected a JSON array in {self._file_path.name}, "
                f"got {type(raw_data).__name__}."
            )

        tweets: list[RawTweet] = []
        for idx, item in enumerate(raw_data):
            try:
                tweet = self._parse_twitter_export_item(item)
                tweets.append(tweet)
            except Exception:
                logger.warning(
                    "Skipping malformed tweet at index %d in %s.",
                    idx,
                    self._file_path.name,
                    exc_info=True,
                )
        return tweets

    def _parse_twitter_export_item(self, item: dict[str, Any]) -> RawTweet:
        """Parse a single item from the Twitter official export JSON.

        The official export wraps each tweet in a ``{"tweet": {...}}`` object.
        Some exports omit the wrapper, so we handle both cases.
        """
        # Handle the wrapper: {"tweet": {…}} or just {…}
        tweet_obj = item.get("tweet", item)

        tweet_id = str(tweet_obj.get("id_str") or tweet_obj.get("id", ""))
        text = tweet_obj.get("full_text") or tweet_obj.get("text", "")
        created_at = self._parse_datetime(tweet_obj.get("created_at", ""))

        # --- Engagement metrics ---
        like_count = _safe_int(tweet_obj.get("favorite_count", 0))
        retweet_count = _safe_int(tweet_obj.get("retweet_count", 0))
        # The official export does not always include reply_count
        reply_count = _safe_int(tweet_obj.get("reply_count", 0))

        # --- Reply detection ---
        in_reply_to_user = tweet_obj.get("in_reply_to_screen_name")
        in_reply_to_status = tweet_obj.get("in_reply_to_status_id_str")
        is_reply = bool(in_reply_to_user or in_reply_to_status)

        # --- Retweet detection ---
        is_retweet = text.startswith("RT @") or "retweeted_status" in tweet_obj

        # --- Quote tweet detection ---
        is_quote = bool(tweet_obj.get("is_quote_status", False))
        quoted_text: str | None = None
        quoted_status = tweet_obj.get("quoted_status")
        if quoted_status:
            quoted_text = quoted_status.get("full_text") or quoted_status.get("text")

        # --- Entities ---
        entities = tweet_obj.get("entities", {})
        hashtags = [
            ht.get("text", "") for ht in entities.get("hashtags", []) if ht.get("text")
        ]
        mentions = [
            m.get("screen_name", "")
            for m in entities.get("user_mentions", [])
            if m.get("screen_name")
        ]
        urls = [
            u.get("expanded_url") or u.get("url", "")
            for u in entities.get("urls", [])
            if u.get("expanded_url") or u.get("url")
        ]

        # --- Media ---
        media_types: list[str] = []
        extended_entities = tweet_obj.get("extended_entities", {})
        for media_obj in extended_entities.get("media", entities.get("media", [])):
            mtype = media_obj.get("type", "")
            if mtype == "photo":
                media_types.append("image")
            elif mtype == "animated_gif":
                media_types.append("gif")
            elif mtype == "video":
                media_types.append("video")
            elif mtype:
                media_types.append(mtype)

        # --- Language ---
        language = tweet_obj.get("lang")

        return RawTweet(
            id=tweet_id,
            text=text,
            created_at=created_at,
            is_reply=is_reply,
            reply_to_user=in_reply_to_user,
            reply_to_tweet_id=in_reply_to_status,
            is_retweet=is_retweet,
            is_quote_tweet=is_quote,
            quoted_text=quoted_text,
            hashtags=hashtags,
            mentions=mentions,
            urls=urls,
            media_types=media_types,
            like_count=like_count,
            retweet_count=retweet_count,
            reply_count=reply_count,
            language=language,
        )

    # --- CSV (simple format) ---

    def _load_csv(self) -> list[RawTweet]:
        """Parse the simple CSV format.

        Expected columns: id, text, created_at, is_reply, reply_to_user,
        like_count, retweet_count, reply_count
        """
        tweets: list[RawTweet] = []

        with open(self._file_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)

            if reader.fieldnames is None:
                raise ValueError(
                    f"CSV file {self._file_path.name} appears to be empty "
                    "or has no header row."
                )

            # Validate required columns
            required_cols = {"id", "text", "created_at"}
            missing = required_cols - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV file {self._file_path.name} is missing required columns: "
                    f"{', '.join(sorted(missing))}. "
                    f"Found columns: {', '.join(reader.fieldnames)}"
                )

            for row_num, row in enumerate(reader, start=2):
                try:
                    tweet = self._parse_csv_row(row)
                    tweets.append(tweet)
                except Exception:
                    logger.warning(
                        "Skipping malformed row %d in %s.",
                        row_num,
                        self._file_path.name,
                        exc_info=True,
                    )

        return tweets

    def _parse_csv_row(self, row: dict[str, str]) -> RawTweet:
        """Parse a single CSV row into a RawTweet."""
        tweet_id = row.get("id", "").strip()
        text = row.get("text", "").strip()
        created_at = self._parse_datetime(row.get("created_at", "").strip())

        is_reply_raw = row.get("is_reply", "").strip().lower()
        is_reply = is_reply_raw in ("true", "1", "yes")

        reply_to_user = row.get("reply_to_user", "").strip() or None

        # Detect retweet from text
        is_retweet = text.startswith("RT @")

        return RawTweet(
            id=tweet_id,
            text=text,
            created_at=created_at,
            is_reply=is_reply,
            reply_to_user=reply_to_user,
            is_retweet=is_retweet,
            like_count=_safe_int(row.get("like_count", "0")),
            retweet_count=_safe_int(row.get("retweet_count", "0")),
            reply_count=_safe_int(row.get("reply_count", "0")),
        )

    # --- Date parsing ---

    @staticmethod
    def _parse_datetime(date_str: str) -> datetime:
        """Attempt to parse a date string using multiple known formats.

        Returns a timezone-aware datetime (defaulting to UTC).
        """
        if not date_str:
            return datetime.now(timezone.utc)

        # Try the Twitter archive format first
        all_formats = [_TWITTER_ARCHIVE_DATE_FORMAT] + _FALLBACK_DATE_FORMATS
        for fmt in all_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        # Last resort: try ISO format parsing (handles many edge cases)
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass

        logger.warning(
            "Unable to parse date '%s'; falling back to current time.", date_str
        )
        return datetime.now(timezone.utc)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _safe_int(value: Any) -> int:
    """Convert a value to int, returning 0 on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
