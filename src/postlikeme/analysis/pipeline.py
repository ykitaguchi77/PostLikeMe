"""Analysis pipeline orchestrator.

Coordinates all analysis stages -- preprocessing, structural analysis,
topic discovery, personality estimation, reply analysis, and example
selection -- into a single async ``analyze`` call that produces a
complete :class:`VoiceProfile`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from postlikeme.analysis.example_selector import ExampleSelector
from postlikeme.analysis.personality_analyzer import PersonalityAnalyzer
from postlikeme.analysis.preprocessor import TweetPreprocessor
from postlikeme.analysis.reply_analyzer import ReplyAnalyzer
from postlikeme.analysis.structural_analyzer import StructuralAnalyzer
from postlikeme.analysis.topic_analyzer import TopicAnalyzer
from postlikeme.models.config import AppConfig
from postlikeme.models.raw_tweet import CleanTweet, RawTweet, UserProfile
from postlikeme.models.voice_profile import VoiceProfile
from postlikeme.utils.constants import MIN_TWEETS_FOR_ANALYSIS, RECOMMENDED_TWEETS

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates the full tweet analysis pipeline.

    Given a list of raw tweets and a user profile, runs every analysis
    stage in sequence and assembles the results into a
    :class:`VoiceProfile`.

    Parameters
    ----------
    config:
        Application configuration (used for data directory paths and
        analysis settings).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        # Instantiate all analyzers
        self._preprocessor = TweetPreprocessor(target_language="en")
        self._structural_analyzer = StructuralAnalyzer()
        self._topic_analyzer = TopicAnalyzer()
        self._personality_analyzer = PersonalityAnalyzer()
        self._reply_analyzer = ReplyAnalyzer()
        self._example_selector = ExampleSelector()

    async def analyze(
        self,
        tweets: list[RawTweet],
        user_profile: UserProfile,
    ) -> VoiceProfile:
        """Run the full analysis pipeline and produce a VoiceProfile.

        Pipeline stages:
        1. Preprocess -- filter, clean, separate originals and replies
        2. Structural analysis -- length, vocabulary, punctuation, capitalization
        3. Topic analysis -- discover topics, compute per-topic sentiment
        4. Personality analysis -- Big Five, formality, humor, tone
        5. Reply analysis -- reply opening patterns, engagement type, tone comparison
        6. Example selection -- select diverse, high-quality examples via MMR
        7. Assemble VoiceProfile

        Parameters
        ----------
        tweets:
            Raw tweets collected from any source.
        user_profile:
            Basic public profile information for the account.

        Returns
        -------
        VoiceProfile
            The fully-assembled voice profile.
        """
        total_count = len(tweets)
        logger.info(
            "Starting analysis pipeline for @%s with %d tweets.",
            user_profile.username,
            total_count,
        )

        # -- Validation warnings --
        if total_count == 0:
            logger.error("No tweets provided for analysis.")
            raise ValueError(
                f"No tweets provided for @{user_profile.username}. "
                "Collect tweets first using 'postlikeme collect'."
            )

        if total_count < MIN_TWEETS_FOR_ANALYSIS:
            logger.warning(
                "Only %d tweets available for @%s (minimum recommended: %d, optimal: %d). "
                "Profile quality may be limited.",
                total_count,
                user_profile.username,
                MIN_TWEETS_FOR_ANALYSIS,
                RECOMMENDED_TWEETS,
            )

        # ==================================================================
        # Stage 1: Preprocessing
        # ==================================================================
        logger.info("[1/7] Preprocessing %d tweets...", total_count)
        originals, replies = self._preprocessor.preprocess(tweets)
        all_clean = originals + replies

        if not all_clean:
            logger.error(
                "All %d tweets were filtered out during preprocessing.", total_count
            )
            raise ValueError(
                f"No usable tweets remain after preprocessing for @{user_profile.username}. "
                "Check that the collected tweets contain actual content."
            )

        logger.info(
            "Preprocessing complete: %d originals, %d replies (%d total).",
            len(originals),
            len(replies),
            len(all_clean),
        )

        # ==================================================================
        # Stage 2: Structural analysis
        # ==================================================================
        logger.info("[2/7] Analysing structural style...")
        structural_style, emoji_profile, hashtag_profile = (
            self._structural_analyzer.analyze_full(all_clean)
        )
        logger.info(
            "Structural analysis complete: avg_tweet_length=%.1f, vocab_richness=%d metrics.",
            structural_style.avg_tweet_length,
            len(structural_style.vocabulary_richness),
        )

        # ==================================================================
        # Stage 3: Topic analysis
        # ==================================================================
        logger.info("[3/7] Discovering topics...")
        topic_profile = self._topic_analyzer.analyze(all_clean)
        logger.info(
            "Topic analysis complete: %d topics discovered.",
            len(topic_profile.topics_ranked),
        )

        # ==================================================================
        # Stage 4: Personality analysis
        # ==================================================================
        logger.info("[4/7] Estimating personality traits...")
        personality_profile = self._personality_analyzer.analyze(
            all_clean, structural_style
        )
        logger.info(
            "Personality analysis complete: formality=%.2f, tone=%s, humor=%s.",
            personality_profile.formality_score,
            personality_profile.tone,
            personality_profile.humor_style,
        )

        # ==================================================================
        # Stage 5: Reply analysis
        # ==================================================================
        logger.info("[5/7] Analysing reply patterns...")
        reply_style = self._reply_analyzer.analyze(replies, originals)
        logger.info(
            "Reply analysis complete: engagement_type=%s, avg_reply_length=%.1f.",
            reply_style.engagement_type,
            reply_style.avg_reply_length,
        )

        # ==================================================================
        # Stage 6: Example selection
        # ==================================================================
        logger.info("[6/7] Selecting representative examples...")
        topic_names = [t.name for t in topic_profile.topics_ranked]
        example_tweets = self._example_selector.select(
            originals=originals,
            replies=replies,
            topics=topic_names,
            target_originals=25,
            target_replies=15,
        )
        logger.info(
            "Example selection complete: %d originals, %d reply pairs.",
            len(example_tweets.originals),
            len(example_tweets.replies),
        )

        # ==================================================================
        # Stage 7: Assemble VoiceProfile
        # ==================================================================
        logger.info("[7/7] Assembling voice profile...")

        # Compute date range from clean tweets
        dates = [t.created_at for t in all_clean]
        date_range_start = min(dates) if dates else None
        date_range_end = max(dates) if dates else None

        # Ensure dates are timezone-aware for storage
        if date_range_start and date_range_start.tzinfo is None:
            date_range_start = date_range_start.replace(tzinfo=timezone.utc)
        if date_range_end and date_range_end.tzinfo is None:
            date_range_end = date_range_end.replace(tzinfo=timezone.utc)

        profile = VoiceProfile(
            # Identity
            username=user_profile.username,
            display_name=user_profile.display_name,
            bio=user_profile.bio,
            # Metadata
            schema_version="1.0.0",
            profile_built_at=datetime.now(timezone.utc),
            tweet_count_analyzed=len(all_clean),
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            # Analysis sections
            structural_style=structural_style,
            emoji_profile=emoji_profile,
            hashtag_profile=hashtag_profile,
            topic_profile=topic_profile,
            personality_profile=personality_profile,
            reply_style=reply_style,
            example_tweets=example_tweets,
            # Raw statistics
            raw_statistics={
                "total_tweets_input": total_count,
                "originals_count": len(originals),
                "replies_count": len(replies),
                "filtered_out": total_count - len(all_clean),
            },
        )

        logger.info(
            "Voice profile for @%s assembled successfully (%d tweets analysed).",
            user_profile.username,
            len(all_clean),
        )

        return profile
