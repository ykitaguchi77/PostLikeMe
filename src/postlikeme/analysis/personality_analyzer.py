"""Personality and tone analysis using linguistic markers.

Computes Big Five personality trait approximations, formality score,
humor detection, and assertiveness from tweet text features.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

import numpy as np

from postlikeme.models.raw_tweet import CleanTweet
from postlikeme.models.voice_profile import (
    BigFiveScores,
    PersonalityProfile,
    StructuralStyle,
)
from postlikeme.utils.constants import (
    ACHIEVEMENT_WORDS,
    ANXIETY_WORDS,
    CERTAINTY_WORDS,
    CONTRACTION_MAP,
    FILLER_WORDS,
    FUNCTION_WORDS,
    SLANG_WORDS,
    SOCIAL_WORDS,
    SWEAR_WORDS,
)
from postlikeme.utils.text import detect_contractions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline statistics for z-score normalisation.
# Approximate Twitter averages derived from large-scale studies.
# (Yarkoni 2010, Pennebaker & King 1999, adapted for social media)
# ---------------------------------------------------------------------------

_BASELINES: dict[str, tuple[float, float]] = {
    # (mean, std)
    "article_ratio": (0.045, 0.015),
    "preposition_ratio": (0.090, 0.025),
    "mtld": (60.0, 20.0),
    "avg_word_length": (4.5, 0.6),
    "formality_score": (0.40, 0.15),
    "personal_pronoun_ratio": (0.070, 0.025),
    "negative_emotion_ratio": (0.025, 0.015),
    "anxiety_word_ratio": (0.005, 0.004),
    "first_person_singular_ratio": (0.045, 0.020),
    "negation_ratio": (0.020, 0.010),
    "positive_emotion_ratio": (0.045, 0.020),
    "social_word_ratio": (0.030, 0.015),
    "tweet_frequency": (3.0, 2.5),
    "swear_word_ratio": (0.008, 0.010),
    "anger_word_ratio": (0.006, 0.005),
    "achievement_word_ratio": (0.008, 0.006),
    "certainty_word_ratio": (0.006, 0.005),
    "filler_word_ratio": (0.040, 0.020),
    # Formality sub-features
    "contraction_ratio": (0.035, 0.020),
    "slang_ratio": (0.015, 0.012),
    "complete_sentence_ratio": (0.50, 0.20),
    "avg_sentence_length": (12.0, 5.0),
    "function_word_ratio": (0.40, 0.08),
}

# Positive emotion words (LIWC-inspired subset)
_POSITIVE_EMOTION_WORDS: set[str] = {
    "love", "loved", "loving", "lovely", "like", "liked", "happy",
    "happiness", "great", "wonderful", "amazing", "awesome", "beautiful",
    "excellent", "fantastic", "brilliant", "good", "best", "better",
    "glad", "pleased", "delighted", "joy", "joyful", "excited",
    "exciting", "fun", "enjoy", "enjoyed", "enjoying", "grateful",
    "thankful", "thank", "thanks", "appreciate", "appreciated",
    "proud", "thrilled", "blessed", "hope", "hopeful", "kind",
    "kindness", "generous", "warm", "sunshine", "perfect", "incredible",
    "outstanding", "superb", "nice", "positive", "optimistic",
    "cheerful", "sweet", "heartwarming", "inspiring", "inspired",
    "magnificent", "phenomenal", "remarkable", "terrific", "splendid",
}

# Negative emotion words
_NEGATIVE_EMOTION_WORDS: set[str] = {
    "hate", "hated", "hating", "angry", "anger", "mad", "furious",
    "sad", "sadness", "depressed", "depressing", "depression",
    "terrible", "horrible", "awful", "worst", "bad", "worse",
    "disgusting", "disgusted", "annoyed", "annoying", "frustrated",
    "frustrating", "frustration", "disappointed", "disappointing",
    "miserable", "unhappy", "painful", "pain", "suffer", "suffering",
    "hurt", "hurting", "lonely", "loneliness", "grief", "grieving",
    "ashamed", "shame", "guilt", "guilty", "regret", "upset",
    "bitter", "resentful", "jealous", "jealousy", "envy", "envious",
    "hostile", "contempt", "despise", "loathe", "detest",
}

# Anger words (subset of negative)
_ANGER_WORDS: set[str] = {
    "angry", "anger", "mad", "furious", "rage", "raging", "outrage",
    "outraged", "outrageous", "infuriating", "infuriated", "livid",
    "pissed", "hostile", "hatred", "hate", "hating", "hated",
    "aggressive", "aggression", "violent", "fury", "wrath",
    "enraged", "bitter", "resentful", "irritated", "irritating",
}

# Negation words
_NEGATION_WORDS: set[str] = {
    "not", "no", "never", "neither", "nor", "nothing", "nowhere",
    "nobody", "none", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't",
    "haven't", "hadn't", "wouldn't", "shouldn't", "couldn't",
    "mustn't", "ain't", "shan't",
}

# Humor markers
_HUMOR_MARKERS: set[str] = {"lol", "lmao", "rofl", "haha", "hahaha", "lmfao"}
_HUMOR_EMOJIS: set[str] = {"\U0001f602", "\U0001f923", "\U0001f480"}  # 😂 🤣 💀

# Hedging / assertive word sets
_HEDGING_PHRASES: list[str] = [
    "i think", "maybe", "perhaps", "probably", "might",
    "kind of", "sort of",
]
_ASSERTIVE_PHRASES: list[str] = [
    "clearly", "obviously", "definitely", "absolutely",
    "certainly", "must", "always", "never",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _z_score(value: float, feature_name: str) -> float:
    """Compute z-score against baseline statistics."""
    mean, std = _BASELINES.get(feature_name, (0.0, 1.0))
    if std <= 0:
        return 0.0
    return (value - mean) / std


def _sigmoid(x: float) -> float:
    """Standard sigmoid function mapping R -> (0, 1)."""
    # Clamp to prevent overflow
    x = max(min(x, 10.0), -10.0)
    return 1.0 / (1.0 + math.exp(-x))


def _count_word_set_ratio(
    tokens: list[str], word_set: set[str], total: int
) -> float:
    """Fraction of tokens that appear in word_set."""
    if total == 0:
        return 0.0
    count = sum(1 for t in tokens if t in word_set)
    return count / total


def _count_phrase_occurrences(text_lower: str, phrases: list[str]) -> int:
    """Count how many times any phrase from the list appears in text."""
    count = 0
    for phrase in phrases:
        count += text_lower.count(phrase)
    return count


# ---------------------------------------------------------------------------
# PersonalityAnalyzer
# ---------------------------------------------------------------------------


class PersonalityAnalyzer:
    """Estimate personality traits from linguistic markers.

    Computes Big Five approximations, formality score, humor indicators,
    and assertiveness using weighted z-score heuristics with sigmoid
    normalization.
    """

    def analyze(
        self,
        tweets: list[CleanTweet],
        structural_style: StructuralStyle,
    ) -> PersonalityProfile:
        """Analyze tweets and structural data to produce a PersonalityProfile."""
        if not tweets:
            logger.warning("No tweets provided to PersonalityAnalyzer.")
            return PersonalityProfile()

        # -- Tokenize all tweets --
        all_tokens: list[str] = []
        all_texts: list[str] = []
        for t in tweets:
            all_texts.append(t.text)
            words = t.text.lower().split()
            all_tokens.extend(words)

        total_tokens = len(all_tokens)
        if total_tokens == 0:
            return PersonalityProfile()

        combined_text = " ".join(all_texts)
        combined_lower = combined_text.lower()

        # ============================================================
        # Feature extraction
        # ============================================================
        features = self._extract_features(
            tweets, all_tokens, total_tokens, combined_lower, structural_style
        )

        # ============================================================
        # Formality score
        # ============================================================
        formality = self._compute_formality(features)

        # Update features with computed formality for openness calculation
        features["formality_score"] = formality

        # ============================================================
        # Big Five
        # ============================================================
        big_five = self._compute_big_five(features)

        # ============================================================
        # Humor detection
        # ============================================================
        humor_style, humor_rate = self._detect_humor(tweets, all_tokens, total_tokens)

        # ============================================================
        # Assertiveness
        # ============================================================
        assertiveness = self._compute_assertiveness(combined_lower)

        # ============================================================
        # Tone classification
        # ============================================================
        tone = self._classify_tone(formality, features)

        # ============================================================
        # Catchphrases (repeated n-grams)
        # ============================================================
        catchphrases = self._detect_catchphrases(all_texts)

        # ============================================================
        # Rhetorical devices
        # ============================================================
        rhetorical = self._detect_rhetorical_devices(tweets, structural_style)

        return PersonalityProfile(
            formality_score=round(formality, 4),
            humor_style=humor_style,
            tone=tone,
            big_five=big_five,
            rhetorical_devices=rhetorical,
            catchphrases=catchphrases,
            assertiveness=round(assertiveness, 4),
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        tweets: list[CleanTweet],
        all_tokens: list[str],
        total_tokens: int,
        combined_lower: str,
        structural_style: StructuralStyle,
    ) -> dict[str, float]:
        """Extract all linguistic features needed for personality scoring."""
        features: dict[str, float] = {}

        # Article ratio (the, a, an)
        articles = {"the", "a", "an"}
        features["article_ratio"] = _count_word_set_ratio(
            all_tokens, articles, total_tokens
        )

        # Preposition ratio (from POS distribution if available)
        pos = structural_style.pos_distribution
        # Use function_word_ratio minus pronouns as proxy for preposition-heavy function words
        # Or we can count common prepositions directly
        prepositions = {
            "in", "on", "at", "to", "for", "with", "from", "by", "about",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "over", "against", "among",
        }
        features["preposition_ratio"] = _count_word_set_ratio(
            all_tokens, prepositions, total_tokens
        )

        # MTLD from vocabulary richness
        features["mtld"] = structural_style.vocabulary_richness.get("mtld", 60.0)

        # Average word length
        word_lengths = [len(t) for t in all_tokens if t.isalpha()]
        features["avg_word_length"] = (
            float(np.mean(word_lengths)) if word_lengths else 4.5
        )

        # Personal pronoun ratio (I, me, my, we, us, our, you, your, he, she, they, etc.)
        personal_pronouns = {
            "i", "me", "my", "mine", "myself",
            "we", "us", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "they", "them", "their", "theirs", "themselves",
        }
        features["personal_pronoun_ratio"] = _count_word_set_ratio(
            all_tokens, personal_pronouns, total_tokens
        )

        # First person singular ratio (I, me, my, mine, myself)
        first_person_sg = {"i", "me", "my", "mine", "myself"}
        features["first_person_singular_ratio"] = _count_word_set_ratio(
            all_tokens, first_person_sg, total_tokens
        )

        # Emotion word ratios
        features["positive_emotion_ratio"] = _count_word_set_ratio(
            all_tokens, _POSITIVE_EMOTION_WORDS, total_tokens
        )
        features["negative_emotion_ratio"] = _count_word_set_ratio(
            all_tokens, _NEGATIVE_EMOTION_WORDS, total_tokens
        )
        features["anxiety_word_ratio"] = _count_word_set_ratio(
            all_tokens, ANXIETY_WORDS, total_tokens
        )
        features["anger_word_ratio"] = _count_word_set_ratio(
            all_tokens, _ANGER_WORDS, total_tokens
        )
        features["negation_ratio"] = _count_word_set_ratio(
            all_tokens, _NEGATION_WORDS, total_tokens
        )

        # Social, achievement, certainty, filler, swear word ratios
        features["social_word_ratio"] = _count_word_set_ratio(
            all_tokens, SOCIAL_WORDS, total_tokens
        )
        features["swear_word_ratio"] = _count_word_set_ratio(
            all_tokens, SWEAR_WORDS, total_tokens
        )
        features["achievement_word_ratio"] = _count_word_set_ratio(
            all_tokens, ACHIEVEMENT_WORDS, total_tokens
        )
        features["certainty_word_ratio"] = _count_word_set_ratio(
            all_tokens, CERTAINTY_WORDS, total_tokens
        )

        # Filler words -- some are multi-word, need phrase matching
        single_fillers = {w for w in FILLER_WORDS if " " not in w}
        multi_fillers = [w for w in FILLER_WORDS if " " in w]
        single_count = sum(1 for t in all_tokens if t in single_fillers)
        multi_count = _count_phrase_occurrences(combined_lower, multi_fillers)
        features["filler_word_ratio"] = (single_count + multi_count) / total_tokens

        # Tweet frequency (tweets per day, using date range)
        if len(tweets) >= 2:
            dates = sorted(t.created_at for t in tweets)
            date_range = (dates[-1] - dates[0]).total_seconds() / 86400
            if date_range > 0:
                features["tweet_frequency"] = len(tweets) / date_range
            else:
                features["tweet_frequency"] = float(len(tweets))
        else:
            features["tweet_frequency"] = 1.0

        # Contraction ratio
        all_text = " ".join(t.text for t in tweets)
        contractions_found = detect_contractions(all_text)
        features["contraction_ratio"] = len(contractions_found) / total_tokens

        # Slang ratio
        features["slang_ratio"] = _count_word_set_ratio(
            all_tokens, SLANG_WORDS, total_tokens
        )

        # Complete sentence ratio (from fragment ratio)
        features["complete_sentence_ratio"] = 1.0 - structural_style.fragment_ratio

        # Average sentence length
        features["avg_sentence_length"] = structural_style.avg_sentence_length

        # Function word ratio
        features["function_word_ratio"] = pos.get("function_word_ratio", 0.40)

        return features

    # ------------------------------------------------------------------
    # Formality score
    # ------------------------------------------------------------------

    def _compute_formality(self, features: dict[str, float]) -> float:
        """Compute formality score using weighted z-scores.

        0.0 = very casual, 1.0 = very formal.
        """
        score = (
            -0.30 * _z_score(features.get("contraction_ratio", 0.035), "contraction_ratio")
            + -0.25 * _z_score(features.get("slang_ratio", 0.015), "slang_ratio")
            + 0.20 * _z_score(features.get("complete_sentence_ratio", 0.50), "complete_sentence_ratio")
            + 0.15 * _z_score(features.get("avg_sentence_length", 12.0), "avg_sentence_length")
            + 0.10 * _z_score(features.get("function_word_ratio", 0.40), "function_word_ratio")
        )
        return _sigmoid(score)

    # ------------------------------------------------------------------
    # Big Five
    # ------------------------------------------------------------------

    def _compute_big_five(self, features: dict[str, float]) -> BigFiveScores:
        """Compute approximate Big Five personality trait scores."""

        # Openness
        openness_raw = (
            0.25 * _z_score(features["article_ratio"], "article_ratio")
            + 0.20 * _z_score(features["preposition_ratio"], "preposition_ratio")
            + 0.25 * _z_score(features["mtld"], "mtld")
            + 0.15 * _z_score(features["avg_word_length"], "avg_word_length")
            + 0.15 * _z_score(features["formality_score"], "formality_score")
            - 0.20 * _z_score(features["personal_pronoun_ratio"], "personal_pronoun_ratio")
        )

        # Neuroticism
        neuroticism_raw = (
            0.30 * _z_score(features["negative_emotion_ratio"], "negative_emotion_ratio")
            + 0.20 * _z_score(features["anxiety_word_ratio"], "anxiety_word_ratio")
            + 0.20 * _z_score(features["first_person_singular_ratio"], "first_person_singular_ratio")
            + 0.15 * _z_score(features["negation_ratio"], "negation_ratio")
            - 0.15 * _z_score(features["positive_emotion_ratio"], "positive_emotion_ratio")
        )

        # Extraversion
        extraversion_raw = (
            0.30 * _z_score(features["positive_emotion_ratio"], "positive_emotion_ratio")
            + 0.30 * _z_score(features["social_word_ratio"], "social_word_ratio")
            + 0.20 * _z_score(features["tweet_frequency"], "tweet_frequency")
            - 0.20 * _z_score(features["negative_emotion_ratio"], "negative_emotion_ratio")
        )

        # Agreeableness
        agreeableness_raw = (
            -0.35 * _z_score(features["swear_word_ratio"], "swear_word_ratio")
            + 0.25 * _z_score(features["positive_emotion_ratio"], "positive_emotion_ratio")
            - 0.20 * _z_score(features["anger_word_ratio"], "anger_word_ratio")
            - 0.20 * _z_score(features["negative_emotion_ratio"], "negative_emotion_ratio")
        )

        # Conscientiousness
        conscientiousness_raw = (
            0.30 * _z_score(features["achievement_word_ratio"], "achievement_word_ratio")
            + 0.25 * _z_score(features["certainty_word_ratio"], "certainty_word_ratio")
            - 0.25 * _z_score(features["filler_word_ratio"], "filler_word_ratio")
            - 0.20 * _z_score(features["swear_word_ratio"], "swear_word_ratio")
        )

        return BigFiveScores(
            openness=round(_sigmoid(openness_raw), 4),
            conscientiousness=round(_sigmoid(conscientiousness_raw), 4),
            extraversion=round(_sigmoid(extraversion_raw), 4),
            agreeableness=round(_sigmoid(agreeableness_raw), 4),
            neuroticism=round(_sigmoid(neuroticism_raw), 4),
        )

    # ------------------------------------------------------------------
    # Humor detection
    # ------------------------------------------------------------------

    def _detect_humor(
        self,
        tweets: list[CleanTweet],
        all_tokens: list[str],
        total_tokens: int,
    ) -> tuple[str, float]:
        """Detect humor style and frequency.

        Returns (humor_style, humor_rate).
        """
        humor_count = 0
        sarcasm_count = 0

        # Count humor markers in tokens
        for token in all_tokens:
            if token in _HUMOR_MARKERS:
                humor_count += 1

        # Count humor emojis in all tweets
        for tweet in tweets:
            for ep in tweet.emoji_positions:
                if ep.emoji in _HUMOR_EMOJIS:
                    humor_count += 1

            # Sarcasm indicators
            text_lower = tweet.text.lower()
            if "/s" in text_lower:
                sarcasm_count += 1
            # "sure" followed by negative context (simple heuristic)
            if re.search(r"\bsure\b.*\b(not|never|no|right)\b", text_lower):
                sarcasm_count += 1

        # Punchline structure: short tweet after a much longer tweet
        punchline_count = 0
        for i in range(1, len(tweets)):
            prev_len = len(tweets[i - 1].text)
            curr_len = len(tweets[i].text)
            if prev_len > 100 and curr_len < 40:
                punchline_count += 1

        total_humor_signals = humor_count + sarcasm_count + punchline_count
        humor_rate = total_humor_signals / max(len(tweets), 1)

        # Classify humor frequency
        if humor_rate > 0.15:
            frequency = "frequent"
        elif humor_rate > 0.05:
            frequency = "occasional"
        else:
            frequency = "rare"

        # Determine style
        if sarcasm_count > humor_count * 0.3 and sarcasm_count > 3:
            humor_style = "sarcastic"
        elif punchline_count > humor_count * 0.3 and punchline_count > 3:
            humor_style = "dry"
        elif humor_count > 0 and frequency != "rare":
            # Check for self-deprecating patterns
            self_deprecating = 0
            for tweet in tweets:
                tl = tweet.text.lower()
                if any(
                    p in tl
                    for p in ["i'm so bad", "i suck", "my dumb", "i'm an idiot", "i'm the worst"]
                ):
                    self_deprecating += 1
            if self_deprecating > 2:
                humor_style = "self_deprecating"
            else:
                humor_style = "wholesome"
        else:
            humor_style = "none"

        return humor_style, humor_rate

    # ------------------------------------------------------------------
    # Assertiveness
    # ------------------------------------------------------------------

    def _compute_assertiveness(self, combined_lower: str) -> float:
        """Compute assertiveness score from hedging vs assertive word usage.

        assertiveness = assertive_count / (assertive_count + hedging_count + 1)
        """
        hedging_count = _count_phrase_occurrences(combined_lower, _HEDGING_PHRASES)
        assertive_count = _count_phrase_occurrences(combined_lower, _ASSERTIVE_PHRASES)

        return assertive_count / (assertive_count + hedging_count + 1)

    # ------------------------------------------------------------------
    # Tone classification
    # ------------------------------------------------------------------

    def _classify_tone(
        self, formality: float, features: dict[str, float]
    ) -> str:
        """Classify the dominant communication tone."""
        if formality > 0.75:
            if features.get("avg_sentence_length", 0) > 18:
                return "academic"
            return "professional"
        elif formality < 0.35:
            return "casual"
        else:
            # Check for provocative or inspirational tones
            if features.get("swear_word_ratio", 0) > 0.015:
                return "provocative"
            if features.get("positive_emotion_ratio", 0) > 0.06:
                return "inspirational"
            return "casual"

    # ------------------------------------------------------------------
    # Catchphrase detection
    # ------------------------------------------------------------------

    def _detect_catchphrases(self, texts: list[str]) -> list[str]:
        """Detect repeated phrases (2-4 grams) that appear unusually often."""
        ngram_counter: Counter = Counter()

        for text in texts:
            words = text.lower().split()
            # Generate 2-grams, 3-grams, 4-grams
            for n in range(2, 5):
                for i in range(len(words) - n + 1):
                    gram = " ".join(words[i : i + n])
                    # Filter out very common patterns
                    if not all(
                        w in {"the", "a", "an", "is", "of", "to", "in", "and", "it", "for"}
                        for w in words[i : i + n]
                    ):
                        ngram_counter[gram] += 1

        # Catchphrase = n-gram appearing in >2% of tweets and at least 5 times
        min_count = max(5, len(texts) // 50)
        catchphrases = [
            gram
            for gram, count in ngram_counter.most_common(100)
            if count >= min_count
        ]

        return catchphrases[:10]

    # ------------------------------------------------------------------
    # Rhetorical device detection
    # ------------------------------------------------------------------

    def _detect_rhetorical_devices(
        self, tweets: list[CleanTweet], structural: StructuralStyle
    ) -> list[str]:
        """Detect commonly used rhetorical devices."""
        devices: list[str] = []

        # Rhetorical questions
        if structural.question_ratio > 0.15:
            devices.append("rhetorical_question")

        # Hyperbole markers
        hyperbole_markers = {
            "literally", "absolutely", "completely", "totally", "insanely",
            "incredibly", "unbelievably", "impossibly", "extremely",
        }
        hyperbole_count = 0
        for tweet in tweets:
            tl = tweet.text.lower()
            if any(m in tl for m in hyperbole_markers):
                hyperbole_count += 1
        if hyperbole_count / max(len(tweets), 1) > 0.05:
            devices.append("hyperbole")

        # Repetition (same word 3+ times in a tweet)
        repetition_count = 0
        for tweet in tweets:
            words = tweet.text.lower().split()
            freq = Counter(words)
            if any(c >= 3 for w, c in freq.items() if len(w) > 2):
                repetition_count += 1
        if repetition_count / max(len(tweets), 1) > 0.03:
            devices.append("repetition")

        # Lists / enumeration (tweets with "1.", "2." or "first", "second")
        list_count = 0
        for tweet in tweets:
            if re.search(r"\b(1\.|2\.|first|second|third)\b", tweet.text.lower()):
                list_count += 1
        if list_count / max(len(tweets), 1) > 0.03:
            devices.append("enumeration")

        # Exclamations / emphasis
        if structural.exclamation_ratio > 0.15:
            devices.append("exclamation")

        # Metaphor / simile detection (simple: "like a", "as a", "is a")
        simile_count = 0
        for tweet in tweets:
            if re.search(r"\b(like a|as a|as if|is a kind of)\b", tweet.text.lower()):
                simile_count += 1
        if simile_count / max(len(tweets), 1) > 0.03:
            devices.append("metaphor")

        # Anaphora (multiple tweets starting with the same word)
        first_words = Counter()
        for tweet in tweets:
            words = tweet.text.split()
            if words:
                first_words[words[0].lower()] += 1
        top_first = first_words.most_common(1)
        if top_first and top_first[0][1] / max(len(tweets), 1) > 0.10:
            devices.append("anaphora")

        return devices
