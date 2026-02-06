"""Reply style analysis.

Compares how an author writes replies vs. original tweets, classifies
opening patterns, determines dominant engagement type, and measures
tone differences.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

import numpy as np

from postlikeme.models.raw_tweet import CleanTweet
from postlikeme.models.voice_profile import ReplyStyle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Opening pattern keywords
# ---------------------------------------------------------------------------

_GREETING_WORDS: set[str] = {
    "hey", "hi", "hello", "yo", "howdy", "sup", "heya", "hiya",
    "good morning", "good evening", "good afternoon", "morning",
}

_AGREEMENT_WORDS: set[str] = {
    "yes", "yeah", "agreed", "exactly", "100%", "true", "right",
    "correct", "absolutely", "totally", "definitely", "facts",
    "this", "same", "yep", "yup", "for sure", "precisely",
}

_DISAGREEMENT_WORDS: set[str] = {
    "no", "nah", "actually", "but", "disagree", "wrong", "false",
    "nope", "not really", "i disagree", "that's not", "incorrect",
    "however", "on the contrary",
}

# Words indicating supportive engagement
_SUPPORTIVE_WORDS: set[str] = {
    "agree", "agreed", "great", "awesome", "love", "exactly", "yes",
    "thanks", "thank", "appreciate", "congrats", "congratulations",
    "well done", "nice", "good point", "true", "right", "absolutely",
    "totally", "100%", "wonderful", "amazing", "brilliant", "perfect",
    "support", "respect",
}

# Words indicating debate / disagreement
_DEBATE_WORDS: set[str] = {
    "disagree", "wrong", "actually", "but", "however", "no",
    "not really", "incorrect", "false", "counterpoint", "on the other hand",
    "i'd argue", "debatable", "questionable", "misleading",
    "that's not", "the problem", "issue",
}

# Humor markers for witty classification
_HUMOR_MARKERS: set[str] = {
    "lol", "lmao", "rofl", "haha", "hahaha", "lmfao", "bruh",
}
_HUMOR_EMOJIS: set[str] = {"\U0001f602", "\U0001f923", "\U0001f480", "\U0001f60f"}


# ---------------------------------------------------------------------------
# ReplyAnalyzer
# ---------------------------------------------------------------------------


class ReplyAnalyzer:
    """Analyze reply behavior comparing replies to original tweets."""

    def analyze(
        self,
        replies: list[CleanTweet],
        originals: list[CleanTweet],
    ) -> ReplyStyle:
        """Analyze reply tweets against original tweets.

        Parameters
        ----------
        replies:
            List of reply tweets.
        originals:
            List of original (non-reply) tweets for comparison.

        Returns
        -------
        ReplyStyle
            Reply behavior profile.
        """
        if not replies:
            logger.info("No reply tweets to analyze.")
            return ReplyStyle()

        # -- Length comparison --
        reply_lengths = [len(r.text) for r in replies]
        original_lengths = [len(o.text) for o in originals] if originals else [1]

        mean_reply_len = float(np.mean(reply_lengths)) if reply_lengths else 0.0
        mean_orig_len = float(np.mean(original_lengths)) if original_lengths else 1.0
        length_ratio = mean_reply_len / mean_orig_len if mean_orig_len > 0 else 1.0

        # -- Opening pattern classification --
        opening_patterns = self._classify_openings(replies)

        # -- Engagement type --
        engagement_type = self._classify_engagement_type(replies)

        # -- Sentiment comparison via VADER --
        sentiment_diff, tone_vs_originals = self._compute_tone_comparison(
            replies, originals
        )

        return ReplyStyle(
            avg_reply_length=round(mean_reply_len, 2),
            opening_patterns=opening_patterns,
            tone_vs_originals=tone_vs_originals,
            engagement_type=engagement_type,
            length_ratio=round(length_ratio, 4),
            sentiment_diff=round(sentiment_diff, 4),
        )

    # ------------------------------------------------------------------
    # Opening pattern classification
    # ------------------------------------------------------------------

    def _classify_openings(self, replies: list[CleanTweet]) -> dict[str, float]:
        """Classify the opening pattern of each reply and return distribution."""
        pattern_counter: Counter = Counter()

        for reply in replies:
            pattern = self._classify_single_opening(reply.text)
            pattern_counter[pattern] += 1

        total = sum(pattern_counter.values())
        if total == 0:
            return {}

        return {
            pattern: round(count / total, 4)
            for pattern, count in pattern_counter.most_common()
        }

    def _classify_single_opening(self, text: str) -> str:
        """Classify a single reply's opening pattern."""
        # Strip leading @USER mentions for analysis
        clean = re.sub(r"^(@USER\s*)+", "", text).strip()
        if not clean:
            return "direct"

        text_lower = clean.lower()
        first_word = text_lower.split()[0] if text_lower.split() else ""

        # Check for emoji start
        try:
            import emoji as emoji_lib

            if emoji_lib.emoji_list(clean):
                first_emoji = emoji_lib.emoji_list(clean)[0]
                if first_emoji["match_start"] == 0:
                    return "emoji_start"
        except ImportError:
            pass

        # Check for quote
        if clean.startswith('"') or clean.startswith(">") or clean.startswith("\u201c"):
            return "quote"

        # Check for greeting
        for greeting in _GREETING_WORDS:
            if text_lower.startswith(greeting):
                return "greeting"

        # Check for agreement
        for agree_word in _AGREEMENT_WORDS:
            if text_lower.startswith(agree_word):
                return "agreement"

        # Check for disagreement
        for disagree_word in _DISAGREEMENT_WORDS:
            if text_lower.startswith(disagree_word):
                return "disagreement"

        # Default: direct response
        return "direct"

    # ------------------------------------------------------------------
    # Engagement type classification
    # ------------------------------------------------------------------

    def _classify_engagement_type(self, replies: list[CleanTweet]) -> str:
        """Classify the dominant engagement type across all replies.

        Types: supportive, debate, witty, informative.
        """
        scores: dict[str, int] = {
            "supportive": 0,
            "debate": 0,
            "witty": 0,
            "informative": 0,
        }

        for reply in replies:
            text_lower = reply.text.lower()
            words = set(text_lower.split())

            # Supportive: positive sentiment + agreement words
            supportive_hits = len(words & _SUPPORTIVE_WORDS)
            if supportive_hits > 0:
                scores["supportive"] += supportive_hits

            # Debate: disagreement words or negative framing
            debate_hits = len(words & _DEBATE_WORDS)
            # Also check multi-word debate phrases
            for phrase in ("that's not", "i'd argue", "on the other hand", "the problem"):
                if phrase in text_lower:
                    debate_hits += 1
            if debate_hits > 0:
                scores["debate"] += debate_hits

            # Witty: humor markers or emojis
            humor_hits = len(words & _HUMOR_MARKERS)
            for ep in reply.emoji_positions:
                if ep.emoji in _HUMOR_EMOJIS:
                    humor_hits += 1
            if humor_hits > 0:
                scores["witty"] += humor_hits

            # Informative: longer replies with links/data or explanatory patterns
            if len(reply.text) > 100:
                scores["informative"] += 1
                # Check for explanatory patterns
                if any(
                    p in text_lower
                    for p in ("because", "the reason", "here's why", "for example",
                              "according to", "research shows", "data shows", "source:")
                ):
                    scores["informative"] += 1

        if not any(scores.values()):
            return "supportive"

        return max(scores, key=lambda k: scores[k])

    # ------------------------------------------------------------------
    # Tone comparison
    # ------------------------------------------------------------------

    def _compute_tone_comparison(
        self,
        replies: list[CleanTweet],
        originals: list[CleanTweet],
    ) -> tuple[float, str]:
        """Compare sentiment between replies and originals using VADER.

        Returns (sentiment_diff, tone_label).
        """
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()

            reply_sentiments = [
                analyzer.polarity_scores(r.text)["compound"] for r in replies
            ]
            original_sentiments = [
                analyzer.polarity_scores(o.text)["compound"] for o in originals
            ] if originals else [0.0]

            mean_reply = float(np.mean(reply_sentiments)) if reply_sentiments else 0.0
            mean_orig = float(np.mean(original_sentiments)) if original_sentiments else 0.0
            diff = mean_reply - mean_orig

            # Determine tone comparison label
            if diff > 0.15:
                tone = "more_casual"  # more positive/casual in replies
            elif diff < -0.15:
                tone = "more_formal"  # more negative/restrained in replies
            else:
                tone = "similar"

            return diff, tone

        except ImportError:
            logger.warning("vaderSentiment not installed; tone comparison unavailable.")
            return 0.0, "similar"
