"""The VoiceProfile model: a complete stylistic fingerprint of an X/Twitter account.

This is the central data structure of PostLikeMe. It captures every measurable
aspect of how a person writes on Twitter -- structural patterns, emoji habits,
topic preferences, personality traits, and reply behaviour -- so the generation
engine can reproduce that voice faithfully.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Sub-profiles
# ---------------------------------------------------------------------------


class TweetLengthDistribution(BaseModel):
    """Statistical distribution of tweet character lengths."""

    mean: float = Field(description="Mean tweet length in characters")
    median: float = Field(description="Median tweet length")
    std: float = Field(description="Standard deviation of tweet lengths")
    min: int = Field(description="Shortest tweet length")
    max: int = Field(description="Longest tweet length")
    p25: float = Field(description="25th percentile")
    p75: float = Field(description="75th percentile")
    histogram: dict[str, int] = Field(
        default_factory=dict,
        description="Histogram buckets mapping range labels to counts, "
        "e.g. {'0-50': 12, '51-100': 45, ...}",
    )


class PunctuationPatterns(BaseModel):
    """Punctuation usage frequencies, normalized per 1000 tokens."""

    period: float = Field(default=0.0, description="Periods per 1000 tokens")
    comma: float = Field(default=0.0, description="Commas per 1000 tokens")
    exclamation: float = Field(default=0.0, description="Exclamation marks per 1000 tokens")
    question: float = Field(default=0.0, description="Question marks per 1000 tokens")
    ellipsis: float = Field(default=0.0, description="Ellipses per tweet (average)")
    em_dash: float = Field(default=0.0, description="Em-dashes per 1000 tokens")
    parentheses: float = Field(default=0.0, description="Parentheses per 1000 tokens")
    quotation: float = Field(default=0.0, description="Quotation marks per 1000 tokens")
    semicolon: float = Field(default=0.0, description="Semicolons per 1000 tokens")
    colon: float = Field(default=0.0, description="Colons per 1000 tokens")


class CapitalizationPatterns(BaseModel):
    """How the author uses capitalization."""

    all_caps_ratio: float = Field(
        default=0.0, description="Fraction of tokens in ALL CAPS (e.g. 'GREAT')"
    )
    initial_caps_ratio: float = Field(
        default=0.0, description="Fraction of sentences starting with a capital letter"
    )
    lowercase_start_ratio: float = Field(
        default=0.0, description="Fraction of sentences starting lowercase"
    )
    mid_caps_ratio: float = Field(
        default=0.0, description="Fraction of tokens with emphatic mid-sentence capitalisation"
    )


class ReadabilityScores(BaseModel):
    """Corpus-level readability metrics computed over all tweets concatenated."""

    flesch_reading_ease: float = Field(
        default=0.0, description="Flesch Reading Ease (0-100, higher = easier)"
    )
    coleman_liau_index: float = Field(
        default=0.0, description="Coleman-Liau Index (grade level)"
    )
    automated_readability_index: float = Field(
        default=0.0, description="ARI (grade level)"
    )
    gunning_fog: float = Field(default=0.0, description="Gunning Fog Index")


class StructuralStyle(BaseModel):
    """Quantitative structural characteristics of the author's writing."""

    avg_tweet_length: float = Field(
        default=0.0, description="Average tweet length in characters"
    )
    tweet_length_distribution: TweetLengthDistribution | None = Field(
        default=None, description="Full statistical distribution of tweet lengths"
    )
    avg_sentence_length: float = Field(
        default=0.0, description="Average sentence length in tokens"
    )
    vocabulary_richness: dict[str, float] = Field(
        default_factory=dict,
        description="Vocabulary richness metrics: ttr, root_ttr, cttr, mtld, mattr, "
        "yules_k, yules_i, vocd_d, hapax_ratio, sichels_s, honores_r, brunets_w",
    )
    top_vocabulary: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Top 50 non-stopword lemmas with their frequencies",
    )
    punctuation_patterns: PunctuationPatterns = Field(
        default_factory=PunctuationPatterns,
        description="Punctuation usage frequency distribution",
    )
    capitalization_patterns: CapitalizationPatterns = Field(
        default_factory=CapitalizationPatterns,
        description="Capitalization usage patterns",
    )
    fragment_ratio: float = Field(
        default=0.0,
        description="Fraction of sentences that are fragments (no main verb / subject)",
    )
    question_ratio: float = Field(
        default=0.0, description="Fraction of sentences ending with '?'"
    )
    exclamation_ratio: float = Field(
        default=0.0, description="Fraction of sentences ending with '!'"
    )
    pos_distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Part-of-speech tag ratios: noun_ratio, verb_ratio, adj_ratio, "
        "adv_ratio, pronoun_ratio, function_word_ratio",
    )
    readability_scores: ReadabilityScores = Field(
        default_factory=ReadabilityScores,
        description="Corpus-level readability metrics",
    )


class EmojiProfile(BaseModel):
    """How the author uses emojis."""

    usage_rate: float = Field(
        default=0.0, description="Average number of emojis per tweet"
    )
    emoji_tweet_ratio: float = Field(
        default=0.0, description="Fraction of tweets containing at least one emoji"
    )
    top_emojis: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Most frequently used emojis with their counts",
    )
    positional_tendency: dict[str, float] = Field(
        default_factory=dict,
        description="Distribution of emoji positions: 'leading', 'inline', 'trailing' "
        "as fractions summing to 1.0",
    )
    emoji_diversity: float = Field(
        default=0.0,
        description="Unique emojis / total emoji occurrences (emoji-level TTR)",
    )
    emoji_sentiment: float = Field(
        default=0.0,
        description="Average sentiment score of used emojis (-1.0 to +1.0)",
    )


class HashtagProfile(BaseModel):
    """How the author uses hashtags."""

    usage_rate: float = Field(
        default=0.0, description="Average number of hashtags per tweet"
    )
    top_hashtags: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Most frequently used hashtags with their counts (top 20)",
    )
    placement_style: str = Field(
        default="mixed",
        description="Dominant placement: 'inline', 'trailing', or 'mixed'",
    )
    casing_style: str = Field(
        default="mixed",
        description="Dominant casing: 'CamelCase', 'lowercase', 'UPPERCASE', or 'mixed'",
    )


class TopicEntry(BaseModel):
    """A single topic with its importance score and representative keywords."""

    name: str = Field(description="Human-readable topic label")
    importance: float = Field(
        ge=0.0, le=1.0, description="Normalized importance score (0-1)"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Representative keywords for this topic"
    )


class OpinionEntry(BaseModel):
    """Sentiment/opinion stance on a particular topic."""

    topic: str = Field(description="Topic name")
    sentiment_mean: float = Field(description="Mean VADER compound score for this topic")
    sentiment_std: float = Field(default=0.0, description="Standard deviation of sentiment")
    label: str = Field(
        description="Qualitative label: 'enthusiastic', 'generally_positive', "
        "'neutral', 'critical', or 'negative'"
    )


class SemanticCluster(BaseModel):
    """A cluster of semantically related tweets discovered via embeddings."""

    cluster_id: int = Field(description="Cluster identifier")
    label: str = Field(default="", description="Auto-generated or LLM-derived cluster label")
    size: int = Field(description="Number of tweets in this cluster")
    keywords: list[str] = Field(
        default_factory=list, description="Representative keywords"
    )
    representative_tweets: list[str] = Field(
        default_factory=list,
        description="Example tweet texts from this cluster (up to 5)",
    )


class TopicProfile(BaseModel):
    """What the author writes about and their stance on each topic."""

    topics_ranked: list[TopicEntry] = Field(
        default_factory=list,
        description="Topics ranked by recency-weighted importance",
    )
    opinion_map: list[OpinionEntry] = Field(
        default_factory=list,
        description="Sentiment/opinion for each identified topic",
    )
    semantic_clusters: list[SemanticCluster] = Field(
        default_factory=list,
        description="Semantic clusters from embedding-based analysis (optional)",
    )


class BigFiveScores(BaseModel):
    """Approximate Big Five personality trait scores derived from linguistic markers.

    Scores are normalized to 0.0-1.0 via sigmoid. These are rough approximations
    based on research correlations (r ~ 0.14-0.30) and should be interpreted as
    qualitative bands (low / moderate / high), not precise measurements.
    """

    openness: float = Field(default=0.5, ge=0.0, le=1.0)
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0)
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0)
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0)
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0)


class PersonalityProfile(BaseModel):
    """Personality and tone characteristics, combining heuristics and LLM assessment."""

    formality_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0.0 = very casual, 1.0 = very formal",
    )
    humor_style: str = Field(
        default="none",
        description="Dominant humor style: 'dry', 'sarcastic', 'absurdist', "
        "'wholesome', 'self_deprecating', or 'none'",
    )
    tone: str = Field(
        default="casual",
        description="Communication tone: 'casual', 'professional', 'academic', "
        "'provocative', or 'inspirational'",
    )
    big_five: BigFiveScores = Field(
        default_factory=BigFiveScores,
        description="Approximate Big Five personality trait scores",
    )
    rhetorical_devices: list[str] = Field(
        default_factory=list,
        description="Frequently used rhetorical devices, e.g. 'metaphor', "
        "'rhetorical_question', 'hyperbole'",
    )
    catchphrases: list[str] = Field(
        default_factory=list,
        description="Signature phrases or verbal tics the author repeats",
    )
    assertiveness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0.0 = very hedging/tentative, 1.0 = very assertive/direct",
    )


class ReplyStyle(BaseModel):
    """How the author behaves when replying to others."""

    avg_reply_length: float = Field(
        default=0.0, description="Average reply length in characters"
    )
    opening_patterns: dict[str, float] = Field(
        default_factory=dict,
        description="Distribution of reply opening types: 'greeting', 'direct', "
        "'quote', 'emoji_lead', 'agreement', 'disagreement'",
    )
    tone_vs_originals: str = Field(
        default="similar",
        description="How reply tone compares to originals: 'more_casual', "
        "'more_formal', 'similar'",
    )
    engagement_type: str = Field(
        default="supportive",
        description="Dominant engagement type: 'supportive', 'debate', 'witty', "
        "or 'informative'",
    )
    length_ratio: float = Field(
        default=1.0,
        description="Mean reply length / mean original tweet length",
    )
    sentiment_diff: float = Field(
        default=0.0,
        description="Mean reply sentiment minus mean original sentiment",
    )


class ExampleTweets(BaseModel):
    """Curated representative tweets for few-shot prompting."""

    originals: list[str] = Field(
        default_factory=list,
        description="20-30 representative original tweets selected via MMR",
    )
    replies: list[tuple[str, str]] = Field(
        default_factory=list,
        description="15-20 (context_tweet, reply) pairs showing reply behaviour",
    )


# ---------------------------------------------------------------------------
# Top-level VoiceProfile
# ---------------------------------------------------------------------------


class VoiceProfile(BaseModel):
    """Complete voice profile for an X/Twitter account.

    This is the central artefact of PostLikeMe's analysis phase. It serialises
    to a single JSON file stored at ``~/.postlikeme/profiles/<username>.json``
    and is consumed by the generation engine to produce style-matched content.
    """

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
        json_schema_extra={
            "title": "PostLikeMe Voice Profile",
            "description": "Complete stylistic fingerprint of an X/Twitter account",
        },
    )

    # --- Identity ---
    username: str = Field(description="Account handle without @ prefix")
    display_name: str = Field(default="", description="Public display name")
    bio: str = Field(default="", description="Account bio")

    # --- Metadata ---
    schema_version: str = Field(
        default="1.0.0",
        description="Profile schema version for forward-compatible migrations",
    )
    profile_built_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this profile was generated",
    )
    tweet_count_analyzed: int = Field(
        default=0, ge=0, description="Number of tweets used to build this profile"
    )
    date_range_start: datetime | None = Field(
        default=None, description="Earliest tweet date in the analysed corpus"
    )
    date_range_end: datetime | None = Field(
        default=None, description="Latest tweet date in the analysed corpus"
    )

    # --- Analysis sections ---
    structural_style: StructuralStyle = Field(
        default_factory=StructuralStyle,
        description="Quantitative structural writing characteristics",
    )
    emoji_profile: EmojiProfile = Field(
        default_factory=EmojiProfile,
        description="Emoji usage patterns",
    )
    hashtag_profile: HashtagProfile = Field(
        default_factory=HashtagProfile,
        description="Hashtag usage patterns",
    )
    topic_profile: TopicProfile = Field(
        default_factory=TopicProfile,
        description="Topics and opinion stances",
    )
    personality_profile: PersonalityProfile = Field(
        default_factory=PersonalityProfile,
        description="Personality, tone, and rhetorical style",
    )
    reply_style: ReplyStyle = Field(
        default_factory=ReplyStyle,
        description="Reply-specific behaviour patterns",
    )
    example_tweets: ExampleTweets = Field(
        default_factory=ExampleTweets,
        description="Curated representative tweets for few-shot generation",
    )

    # --- Raw statistics ---
    raw_statistics: dict[str, Any] = Field(
        default_factory=dict,
        description="Unstructured bag of additional statistics for debugging "
        "and advanced prompt tuning",
    )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the profile to a JSON string.

        Uses Pydantic v2's ``model_dump`` with ``mode="json"`` so that
        datetimes, tuples, and other non-JSON-native types are properly
        converted.
        """
        data = self.model_dump(mode="json")
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> VoiceProfile:
        """Deserialise a VoiceProfile from a JSON string."""
        data = json.loads(json_str)
        return cls.model_validate(data)

    def to_json_file(self, path: str | object) -> None:
        """Write the profile to a JSON file at *path*.

        *path* can be a ``str`` or ``pathlib.Path``.
        """
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | object) -> VoiceProfile:
        """Read a VoiceProfile from a JSON file at *path*."""
        from pathlib import Path

        target = Path(path)
        return cls.from_json(target.read_text(encoding="utf-8"))
