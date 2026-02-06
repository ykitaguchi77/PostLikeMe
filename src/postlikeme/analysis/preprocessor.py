"""Tweet preprocessing pipeline.

Implements the preprocessing stage (Section 1) of the analysis pipeline as
defined in ``docs/analysis-algorithms.md``.  Converts :class:`RawTweet` objects
into :class:`CleanTweet` objects by filtering, cleaning, and separating tweets
into originals and replies.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from postlikeme.models.raw_tweet import CleanTweet, EmojiPosition, RawTweet
from postlikeme.utils.text import (
    extract_emojis,
    normalize_text,
    replace_mentions_with_placeholder,
    strip_mentions,
    strip_urls,
)

logger = logging.getLogger(__name__)

# Pattern to detect tweets that are only URLs (after stripping whitespace)
_ONLY_URLS_PATTERN = re.compile(
    r"^\s*(https?://\S+\s*)+$", re.IGNORECASE
)

# Pattern to detect tweets that are only @mentions (after stripping whitespace)
_ONLY_MENTIONS_PATTERN = re.compile(
    r"^\s*(@\w{1,15}\s*)+$"
)

# Whitespace normalisation
_MULTI_SPACE = re.compile(r"\s+")

# t.co URL pattern for stripping
_TCO_PATTERN = re.compile(r"https?://t\.co/\w+", re.IGNORECASE)


class TweetPreprocessor:
    """Preprocess raw tweets into cleaned, analysis-ready form.

    The preprocessing pipeline performs:

    1. **Filtering** -- removes pure retweets, URL-only tweets, and
       mention-only tweets.
    2. **Cleaning** -- strips t.co URLs, applies Unicode NFC normalisation,
       tags emoji positions, normalises @mentions to ``@USER``, and
       collapses whitespace.
    3. **Separation** -- splits the cleaned tweets into originals (standalone
       tweets) and replies.

    Parameters
    ----------
    target_language:
        If provided, only tweets whose ``language`` field matches this
        BCP-47 code (e.g. ``"en"``) are retained.  Tweets with
        ``language=None`` are always kept (language unknown).
        Set to ``None`` (default) to skip language filtering.

    Example
    -------
    >>> preprocessor = TweetPreprocessor(target_language="en")
    >>> originals, replies = preprocessor.preprocess(raw_tweets)
    """

    def __init__(self, target_language: str | None = None) -> None:
        self._target_language = target_language

    def preprocess(
        self, tweets: list[RawTweet]
    ) -> tuple[list[CleanTweet], list[CleanTweet]]:
        """Run the full preprocessing pipeline.

        Parameters
        ----------
        tweets:
            Raw tweets collected from any source.

        Returns
        -------
        tuple[list[CleanTweet], list[CleanTweet]]
            A 2-tuple of ``(originals, replies)`` where:
            - ``originals`` are standalone tweets (``is_reply=False``)
            - ``replies`` are reply tweets (``is_reply=True``)

            Both lists are sorted in reverse-chronological order.
        """
        total_input = len(tweets)

        # Step 1: Filter
        filtered = self._filter(tweets)
        filtered_count = total_input - len(filtered)

        # Step 2: Clean and convert to CleanTweet
        cleaned: list[CleanTweet] = []
        for raw_tweet in filtered:
            clean = self._clean(raw_tweet)
            if clean is not None:
                cleaned.append(clean)

        # Step 3: Separate into originals and replies
        originals = [t for t in cleaned if not t.is_reply]
        replies = [t for t in cleaned if t.is_reply]

        # Sort newest first
        originals.sort(key=lambda t: t.created_at, reverse=True)
        replies.sort(key=lambda t: t.created_at, reverse=True)

        logger.info(
            "Preprocessing complete: %d input -> %d filtered out -> "
            "%d originals + %d replies = %d total.",
            total_input,
            filtered_count,
            len(originals),
            len(replies),
            len(originals) + len(replies),
        )

        return originals, replies

    # ------------------------------------------------------------------
    # Step 1: Filtering
    # ------------------------------------------------------------------

    def _filter(self, tweets: list[RawTweet]) -> list[RawTweet]:
        """Apply all filtering rules and return the tweets that pass.

        Filtering rules (from analysis-algorithms.md Section 1.1):
        - Remove pure retweets (``is_retweet == True``)
        - Remove tweets that are only URLs
        - Remove tweets that are only mentions
        - Remove tweets whose language does not match the target (if set)
        """
        result: list[RawTweet] = []

        for tweet in tweets:
            # Rule 1: Remove pure retweets
            if tweet.is_retweet:
                continue

            # Rule 2: Remove URL-only tweets
            if self._is_url_only(tweet.text):
                continue

            # Rule 3: Remove mention-only tweets
            if self._is_mention_only(tweet.text):
                continue

            # Rule 4: Language filter (if target language is specified)
            if self._target_language is not None and tweet.language is not None:
                if tweet.language != self._target_language:
                    continue

            result.append(tweet)

        return result

    @staticmethod
    def _is_url_only(text: str) -> bool:
        """Return True if the tweet text contains only URLs and whitespace."""
        stripped = text.strip()
        if not stripped:
            return True
        return bool(_ONLY_URLS_PATTERN.fullmatch(stripped))

    @staticmethod
    def _is_mention_only(text: str) -> bool:
        """Return True if the tweet text contains only @mentions and whitespace."""
        stripped = text.strip()
        if not stripped:
            return True
        return bool(_ONLY_MENTIONS_PATTERN.fullmatch(stripped))

    # ------------------------------------------------------------------
    # Step 2: Cleaning
    # ------------------------------------------------------------------

    def _clean(self, tweet: RawTweet) -> CleanTweet | None:
        """Apply the full cleaning pipeline to a single tweet.

        Cleaning steps (from analysis-algorithms.md Section 1.2):
        1. Strip t.co URLs
        2. Unicode normalisation (NFC)
        3. Tag emoji positions (preserving emojis in text)
        4. Normalise user mentions to @USER (keeping originals)
        5. Normalise whitespace

        Returns ``None`` if the cleaned text is empty (e.g. the tweet was
        only URLs and mentions).
        """
        text = tweet.text

        # Step 1: Strip t.co URLs
        text = _TCO_PATTERN.sub("", text)

        # Step 2: Unicode NFC normalisation
        text = unicodedata.normalize("NFC", text)

        # Step 3: Tag emoji positions (on the text AFTER URL stripping but
        #         BEFORE mention normalisation, so positions are accurate)
        emoji_data = extract_emojis(text)
        emoji_positions = [
            EmojiPosition(
                emoji=e["emoji"],
                match_start=e["match_start"],
                match_end=e["match_end"],
                position_ratio=e["position_ratio"],
            )
            for e in emoji_data
        ]

        # Step 4: Normalise user mentions to @USER, preserve originals
        text, original_mentions = replace_mentions_with_placeholder(text)

        # Step 5: Normalise whitespace
        text = _MULTI_SPACE.sub(" ", text).strip()

        # Skip tweets that became empty after cleaning
        if not text:
            return None

        return CleanTweet(
            id=tweet.id,
            text=text,
            original_text=tweet.text,
            created_at=tweet.created_at,
            is_reply=tweet.is_reply,
            reply_to_user=tweet.reply_to_user,
            reply_to_tweet_id=tweet.reply_to_tweet_id,
            emoji_positions=emoji_positions,
            original_mentions=original_mentions,
            hashtags=tweet.hashtags,
            like_count=tweet.like_count,
            retweet_count=tweet.retweet_count,
            reply_count=tweet.reply_count,
            language=tweet.language,
            is_quote_tweet=tweet.is_quote_tweet,
            quoted_text=tweet.quoted_text,
        )
