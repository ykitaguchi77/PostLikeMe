"""Twscrape-based collector for scraping X/Twitter without official API keys.

Implements :class:`BaseCollector` using the ``twscrape`` library, which is an
optional dependency (install with ``pip install postlikeme[scraping]``).

This collector is intended as a fallback when the user does not have access
to the official X/Twitter API v2.  It provides graceful degradation with
clear error messages when ``twscrape`` is not installed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from postlikeme.collectors.base import BaseCollector
from postlikeme.models.raw_tweet import RawTweet, UserProfile

logger = logging.getLogger(__name__)

# Default location for the twscrape account pool database
_DEFAULT_DB_PATH = Path.home() / ".postlikeme" / "twscrape_accounts.db"

# Try to import twscrape; set a flag so we can give helpful error messages
try:
    import twscrape
    from twscrape import API as TwscrapeAPI

    _TWSCRAPE_AVAILABLE = True
except ImportError:
    _TWSCRAPE_AVAILABLE = False
    TwscrapeAPI = None  # type: ignore[assignment, misc]


def _require_twscrape() -> None:
    """Raise a clear error if twscrape is not installed."""
    if not _TWSCRAPE_AVAILABLE:
        raise ImportError(
            "The 'twscrape' library is required for scraping-based collection "
            "but is not installed.\n\n"
            "Install it with:\n"
            "  pip install postlikeme[scraping]\n\n"
            "Or install twscrape directly:\n"
            "  pip install twscrape"
        )


class TwscrapeCollector(BaseCollector):
    """Collect tweets by scraping X/Twitter using twscrape.

    This collector requires the optional ``twscrape`` dependency and a
    configured account pool database.

    Parameters
    ----------
    account_pool_db:
        Path to the twscrape SQLite account pool database.
        Defaults to ``~/.postlikeme/twscrape_accounts.db``.

    Raises
    ------
    ImportError
        If ``twscrape`` is not installed.
    """

    def __init__(
        self,
        account_pool_db: Path | str | None = None,
    ) -> None:
        _require_twscrape()

        self._db_path = Path(account_pool_db) if account_pool_db else _DEFAULT_DB_PATH
        # Ensure the parent directory exists for the DB file
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._api: Any = TwscrapeAPI(str(self._db_path))

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    async def collect_user_tweets(
        self,
        username: str,
        count: int = 500,
        include_replies: bool = True,
    ) -> list[RawTweet]:
        """Collect up to *count* recent tweets for *username* via scraping.

        Uses the twscrape search API to find tweets from the user.

        Parameters
        ----------
        username:
            Twitter handle without ``@``.
        count:
            Maximum number of tweets to retrieve.
        include_replies:
            Whether to include reply tweets.

        Returns
        -------
        list[RawTweet]
            Tweets in reverse-chronological order.
        """
        _require_twscrape()

        # Build the search query to find tweets from this user
        query = f"from:{username}"
        if not include_replies:
            query += " -filter:replies"

        tweets: list[RawTweet] = []

        try:
            async for tweet_data in self._api.search(query, limit=count):
                raw_tweet = self._parse_twscrape_tweet(tweet_data)
                tweets.append(raw_tweet)
                if len(tweets) >= count:
                    break
        except Exception as exc:
            logger.error(
                "Error scraping tweets for @%s: %s. "
                "Ensure your twscrape account pool is properly configured. "
                "Run: twscrape add_accounts <file> to add accounts.",
                username,
                exc,
            )
            if not tweets:
                raise LookupError(
                    f"Failed to scrape tweets for @{username}: {exc}. "
                    "Make sure your twscrape account pool is configured. "
                    "See: https://github.com/vladkens/twscrape#readme"
                ) from exc

        # Sort newest first
        tweets.sort(key=lambda t: t.created_at, reverse=True)
        logger.info(
            "Scraped %d tweets for @%s via twscrape.", len(tweets), username
        )
        return tweets

    async def collect_tweet_by_id(self, tweet_id: str) -> RawTweet:
        """Fetch a single tweet by its ID via scraping.

        Parameters
        ----------
        tweet_id:
            Numeric tweet identifier as a string.

        Returns
        -------
        RawTweet

        Raises
        ------
        LookupError
            If the tweet cannot be found or accessed.
        """
        _require_twscrape()

        try:
            tweet_data = await self._api.tweet_details(int(tweet_id))
        except Exception as exc:
            raise LookupError(
                f"Cannot retrieve tweet {tweet_id} via twscrape: {exc}"
            ) from exc

        if tweet_data is None:
            raise LookupError(
                f"Tweet {tweet_id} not found or not accessible via twscrape."
            )

        return self._parse_twscrape_tweet(tweet_data)

    async def get_user_profile(self, username: str) -> UserProfile:
        """Fetch public profile information for *username* via scraping.

        Parameters
        ----------
        username:
            Twitter handle without ``@``.

        Returns
        -------
        UserProfile

        Raises
        ------
        LookupError
            If the user does not exist or cannot be accessed.
        """
        _require_twscrape()

        try:
            user_data = await self._api.user_by_login(username)
        except Exception as exc:
            raise LookupError(
                f"Cannot retrieve profile for @{username} via twscrape: {exc}"
            ) from exc

        if user_data is None:
            raise LookupError(
                f"User @{username} not found via twscrape."
            )

        return UserProfile(
            username=getattr(user_data, "username", username),
            display_name=getattr(user_data, "displayname", username),
            bio=getattr(user_data, "rawDescription", "") or "",
            followers_count=getattr(user_data, "followersCount", 0) or 0,
            following_count=getattr(user_data, "friendsCount", 0) or 0,
            tweet_count=getattr(user_data, "statusesCount", 0) or 0,
        )

    # ------------------------------------------------------------------
    # Internal: parsing twscrape tweet objects
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_twscrape_tweet(tweet_data: Any) -> RawTweet:
        """Convert a twscrape tweet object into a RawTweet model.

        twscrape tweet objects have attributes like:
        - id, id_str
        - rawContent (full tweet text)
        - date (datetime)
        - likeCount, retweetCount, replyCount
        - inReplyToTweetId, inReplyToUser
        - hashtags, mentionedUsers
        - links, media
        - retweetedTweet, quotedTweet
        - lang
        """
        tweet_id = str(getattr(tweet_data, "id", ""))
        text = getattr(tweet_data, "rawContent", "") or getattr(tweet_data, "content", "") or ""
        created_at = getattr(tweet_data, "date", None)
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        elif created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        # --- Engagement metrics ---
        like_count = getattr(tweet_data, "likeCount", 0) or 0
        retweet_count = getattr(tweet_data, "retweetCount", 0) or 0
        reply_count = getattr(tweet_data, "replyCount", 0) or 0

        # --- Reply detection ---
        in_reply_to_tweet = getattr(tweet_data, "inReplyToTweetId", None)
        in_reply_to_user_obj = getattr(tweet_data, "inReplyToUser", None)
        reply_to_user: str | None = None
        if in_reply_to_user_obj is not None:
            reply_to_user = getattr(in_reply_to_user_obj, "username", None)
        is_reply = in_reply_to_tweet is not None or reply_to_user is not None

        reply_to_tweet_id: str | None = None
        if in_reply_to_tweet is not None:
            reply_to_tweet_id = str(in_reply_to_tweet)

        # --- Retweet detection ---
        retweeted_tweet = getattr(tweet_data, "retweetedTweet", None)
        is_retweet = retweeted_tweet is not None

        # --- Quote tweet detection ---
        quoted_tweet = getattr(tweet_data, "quotedTweet", None)
        is_quote_tweet = quoted_tweet is not None
        quoted_text: str | None = None
        if quoted_tweet is not None:
            quoted_text = (
                getattr(quoted_tweet, "rawContent", None)
                or getattr(quoted_tweet, "content", None)
            )

        # --- Hashtags ---
        raw_hashtags = getattr(tweet_data, "hashtags", None) or []
        hashtags: list[str] = []
        for ht in raw_hashtags:
            if isinstance(ht, str):
                hashtags.append(ht.lstrip("#"))
            else:
                tag = getattr(ht, "text", None) or getattr(ht, "tag", None) or str(ht)
                hashtags.append(tag.lstrip("#"))

        # --- Mentions ---
        raw_mentions = getattr(tweet_data, "mentionedUsers", None) or []
        mentions: list[str] = []
        for m in raw_mentions:
            if isinstance(m, str):
                mentions.append(m.lstrip("@"))
            elif m is not None:
                uname = getattr(m, "username", None) or str(m)
                mentions.append(uname.lstrip("@"))

        # --- URLs ---
        raw_links = getattr(tweet_data, "links", None) or []
        urls: list[str] = []
        for link in raw_links:
            if isinstance(link, str):
                urls.append(link)
            else:
                url_str = getattr(link, "url", None) or str(link)
                urls.append(url_str)

        # --- Media ---
        raw_media = getattr(tweet_data, "media", None) or []
        media_types: list[str] = []
        for media_obj in raw_media:
            if media_obj is None:
                continue
            mtype = getattr(media_obj, "type", None) or ""
            if isinstance(mtype, str):
                mtype_lower = mtype.lower()
                if "photo" in mtype_lower or "image" in mtype_lower:
                    media_types.append("image")
                elif "gif" in mtype_lower:
                    media_types.append("gif")
                elif "video" in mtype_lower:
                    media_types.append("video")
                elif mtype:
                    media_types.append(mtype)

        # --- Language ---
        language = getattr(tweet_data, "lang", None)

        return RawTweet(
            id=tweet_id,
            text=text,
            created_at=created_at,
            is_reply=is_reply,
            reply_to_user=reply_to_user,
            reply_to_tweet_id=reply_to_tweet_id,
            is_retweet=is_retweet,
            is_quote_tweet=is_quote_tweet,
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
