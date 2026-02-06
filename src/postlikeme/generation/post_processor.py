"""Post-processing and quality scoring for LLM-generated tweets.

Cleans up common LLM output artefacts, validates length constraints, and
scores how well generated text adheres to the target voice profile.
"""

from __future__ import annotations

import logging
import math
import re

import emoji as emoji_lib

from postlikeme.models.voice_profile import VoiceProfile
from postlikeme.utils.constants import MAX_TWEET_LENGTH

logger = logging.getLogger(__name__)

# Common LLM artefact patterns to strip
_QUOTE_WRAPPERS = re.compile(r'^[\"\u201c\u201d\u2018\u2019\']+|[\"\u201c\u201d\u2018\u2019\']+$')
_LABEL_PREFIX = re.compile(
    r"^(?:(?:here(?:'s| is) (?:a |my |the )?)?(?:tweet|reply|response|post|a tweet|new tweet)"
    r"[:\-\s]*|(?:tweet|reply|response|post)\s*(?:\d+)?[:\-]\s*)",
    re.IGNORECASE,
)
_EXPLANATION_SUFFIX = re.compile(
    r"\s*(?:\(this (?:tweet|reply|post).*?\)|"
    r"\[note:.*?\]|"
    r"\*note:.*?\*|"
    r"---\s*(?:this|note|i ).*?$)",
    re.IGNORECASE | re.DOTALL,
)
_NUMBERING_PREFIX = re.compile(r"^\d+[\.\)]\s*")
_DASH_PREFIX = re.compile(r"^[\-\u2013\u2014]\s+")


class PostProcessor:
    """Cleans, validates, and scores LLM-generated tweet text."""

    def validate_length(self, text: str, max_length: int = MAX_TWEET_LENGTH) -> bool:
        """Check whether *text* fits within the character limit.

        Parameters
        ----------
        text:
            The tweet text to validate.
        max_length:
            Maximum allowed character count.

        Returns
        -------
        bool
            ``True`` if the text is within the limit and non-empty.
        """
        return 0 < len(text.strip()) <= max_length

    def clean_llm_output(self, text: str) -> str:
        """Remove common LLM artefacts from generated text.

        Strips surrounding quotes, "Here's a tweet:" prefixes, trailing
        explanations, numbering, and excessive whitespace.

        Parameters
        ----------
        text:
            Raw LLM output text.

        Returns
        -------
        str
            Cleaned text ready for validation and scoring.
        """
        if not text:
            return ""

        cleaned = text.strip()

        # Remove leading/trailing quotation marks (including smart quotes)
        cleaned = _QUOTE_WRAPPERS.sub("", cleaned).strip()

        # Remove common label prefixes like "Tweet:" or "Here's a tweet:"
        cleaned = _LABEL_PREFIX.sub("", cleaned).strip()

        # Remove numbered list prefixes like "1." or "2)"
        cleaned = _NUMBERING_PREFIX.sub("", cleaned).strip()

        # Remove leading dashes
        cleaned = _DASH_PREFIX.sub("", cleaned).strip()

        # Remove trailing explanations / notes
        cleaned = _EXPLANATION_SUFFIX.sub("", cleaned).strip()

        # Remove any remaining wrapping quotes after other cleaning
        cleaned = _QUOTE_WRAPPERS.sub("", cleaned).strip()

        # Collapse multiple spaces and normalize whitespace
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def score_style_adherence(self, text: str, profile: VoiceProfile) -> float:
        """Score how well *text* matches the target voice profile's style.

        Returns a score between 0.0 (poor match) and 1.0 (excellent match)
        based on multiple style dimensions:
        - Tweet length similarity
        - Emoji usage consistency
        - Hashtag usage consistency
        - Capitalization pattern similarity
        - Question/exclamation pattern alignment

        Parameters
        ----------
        text:
            The generated tweet text to evaluate.
        profile:
            The target voice profile to compare against.

        Returns
        -------
        float
            Style adherence score between 0.0 and 1.0.
        """
        if not text.strip():
            return 0.0

        scores: list[float] = []
        weights: list[float] = []

        # --- Length similarity (weight: 3) ---
        target_len = profile.structural_style.avg_tweet_length
        if target_len > 0:
            len_ratio = len(text) / target_len
            # Gaussian-like penalty: score 1.0 at ratio=1.0, drops for deviation
            len_score = math.exp(-2.0 * (len_ratio - 1.0) ** 2)
            scores.append(len_score)
            weights.append(3.0)

        # --- Emoji usage consistency (weight: 2) ---
        emoji_count = len(emoji_lib.emoji_list(text))
        target_emoji_rate = profile.emoji_profile.usage_rate
        if target_emoji_rate < 0.1:
            # Profile rarely uses emojis; penalize if we generated any
            emoji_score = 1.0 if emoji_count == 0 else max(0.0, 1.0 - emoji_count * 0.3)
        else:
            # Profile uses emojis; score similarity to target rate
            if emoji_count == 0:
                emoji_score = max(0.0, 1.0 - target_emoji_rate * 0.5)
            else:
                ratio = emoji_count / target_emoji_rate
                emoji_score = math.exp(-1.5 * (ratio - 1.0) ** 2)
        scores.append(emoji_score)
        weights.append(2.0)

        # --- Hashtag usage consistency (weight: 1.5) ---
        hashtag_count = len(re.findall(r"#\w+", text))
        target_hashtag_rate = profile.hashtag_profile.usage_rate
        if target_hashtag_rate < 0.05:
            hashtag_score = 1.0 if hashtag_count == 0 else max(0.0, 1.0 - hashtag_count * 0.3)
        else:
            if hashtag_count == 0:
                hashtag_score = max(0.0, 1.0 - target_hashtag_rate * 0.3)
            else:
                ratio = hashtag_count / target_hashtag_rate
                hashtag_score = math.exp(-1.5 * (ratio - 1.0) ** 2)
        scores.append(hashtag_score)
        weights.append(1.5)

        # --- Capitalization pattern (weight: 1) ---
        words = text.split()
        if words:
            all_caps_count = sum(
                1 for w in words if len(w) >= 2 and w.isalpha() and w.isupper()
            )
            all_caps_ratio = all_caps_count / len(words)
            target_caps = profile.structural_style.capitalization_patterns.all_caps_ratio
            caps_diff = abs(all_caps_ratio - target_caps)
            caps_score = max(0.0, 1.0 - caps_diff * 5.0)
            scores.append(caps_score)
            weights.append(1.0)

        # --- Question mark usage (weight: 1) ---
        has_question = "?" in text
        target_question_ratio = profile.structural_style.question_ratio
        if target_question_ratio > 0.2:
            # Profile asks lots of questions; reward questions
            question_score = 0.8 if has_question else 0.4
        elif target_question_ratio < 0.05:
            # Profile rarely asks questions; slight penalty for questions
            question_score = 0.6 if has_question else 0.9
        else:
            question_score = 0.8  # Neutral
        scores.append(question_score)
        weights.append(1.0)

        # --- Exclamation mark usage (weight: 1) ---
        has_exclamation = "!" in text
        target_exclamation_ratio = profile.structural_style.exclamation_ratio
        if target_exclamation_ratio > 0.2:
            exclamation_score = 0.8 if has_exclamation else 0.4
        elif target_exclamation_ratio < 0.05:
            exclamation_score = 0.6 if has_exclamation else 0.9
        else:
            exclamation_score = 0.8
        scores.append(exclamation_score)
        weights.append(1.0)

        # --- Length validity bonus (weight: 2) ---
        if self.validate_length(text):
            scores.append(1.0)
        else:
            scores.append(0.0)
        weights.append(2.0)

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        final_score = weighted_sum / total_weight

        return round(min(1.0, max(0.0, final_score)), 4)

    def rank_candidates(
        self,
        candidates: list[str],
        profile: VoiceProfile,
    ) -> list[tuple[str, float]]:
        """Score and rank candidate tweets by style adherence.

        Parameters
        ----------
        candidates:
            List of generated tweet texts to rank.
        profile:
            The target voice profile for scoring.

        Returns
        -------
        list[tuple[str, float]]
            Candidates paired with their scores, sorted descending by score.
        """
        scored = [
            (text, self.score_style_adherence(text, profile))
            for text in candidates
            if text.strip()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
