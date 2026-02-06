"""JSON-based storage for VoiceProfile files.

Profiles are stored as one JSON file per username at::

    <data_dir>/profiles/<username>.json

This module provides CRUD operations, listing, and export in multiple
formats (JSON, Markdown summary).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from postlikeme.models.voice_profile import VoiceProfile


class ProfileStore:
    """Manages reading and writing VoiceProfile JSON files on disk.

    Parameters
    ----------
    data_dir:
        Root PostLikeMe data directory (typically ``~/.postlikeme``).
        A ``profiles/`` subdirectory will be created automatically.
    """

    def __init__(self, data_dir: Path) -> None:
        self._profiles_dir = data_dir / "profiles"
        self._profiles_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profile_path(self, username: str) -> Path:
        """Return the canonical file path for a given username."""
        safe_name = username.lower().strip().lstrip("@")
        return self._profiles_dir / f"{safe_name}.json"

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def save_profile(self, profile: VoiceProfile) -> Path:
        """Persist a VoiceProfile to disk as JSON.

        If a profile for this username already exists it will be overwritten.

        Parameters
        ----------
        profile:
            The voice profile to save.

        Returns
        -------
        Path
            The absolute path to the written JSON file.
        """
        path = self._profile_path(profile.username)
        path.write_text(profile.to_json(), encoding="utf-8")
        # Restrict file permissions (profile may contain style data the
        # user considers private).
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return path

    def load_profile(self, username: str) -> VoiceProfile | None:
        """Load a VoiceProfile from disk.

        Parameters
        ----------
        username:
            Account handle (with or without ``@`` prefix).

        Returns
        -------
        VoiceProfile | None
            The deserialised profile, or ``None`` if no profile exists
            for this username.
        """
        path = self._profile_path(username)
        if not path.exists():
            return None
        return VoiceProfile.from_json(path.read_text(encoding="utf-8"))

    def list_profiles(self) -> list[str]:
        """Return a sorted list of usernames that have stored profiles.

        Returns
        -------
        list[str]
            Usernames (lowercase, without ``@``).
        """
        profiles: list[str] = []
        for path in sorted(self._profiles_dir.glob("*.json")):
            profiles.append(path.stem)
        return profiles

    def delete_profile(self, username: str) -> bool:
        """Delete the stored profile for *username*.

        Parameters
        ----------
        username:
            Account handle.

        Returns
        -------
        bool
            ``True`` if a profile was deleted, ``False`` if none existed.
        """
        path = self._profile_path(username)
        if path.exists():
            path.unlink()
            return True
        return False

    def export_profile(self, username: str, format: str = "json") -> str:
        """Export a profile in the requested format.

        Parameters
        ----------
        username:
            Account handle.
        format:
            ``"json"`` for raw JSON, ``"markdown"`` for a human-readable
            Markdown summary.

        Returns
        -------
        str
            The profile content as a string in the requested format.

        Raises
        ------
        FileNotFoundError
            If no profile exists for this username.
        ValueError
            If *format* is not ``"json"`` or ``"markdown"``.
        """
        profile = self.load_profile(username)
        if profile is None:
            raise FileNotFoundError(
                f"No profile found for @{username}. "
                f"Run 'postlikeme analyze {username}' first."
            )

        if format == "json":
            return profile.to_json()
        elif format == "markdown":
            return self._render_markdown(profile)
        else:
            raise ValueError(
                f"Unsupported export format: {format!r}. Choose 'json' or 'markdown'."
            )

    # ------------------------------------------------------------------
    # Markdown renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _render_markdown(profile: VoiceProfile) -> str:
        """Render a VoiceProfile as a human-readable Markdown document."""
        lines: list[str] = []

        def _add(text: str = "") -> None:
            lines.append(text)

        _add(f"# Voice Profile: @{profile.username}")
        _add()

        # -- Metadata --
        _add("## Metadata")
        _add()
        _add(f"- **Display name**: {profile.display_name}")
        _add(f"- **Bio**: {profile.bio}")
        _add(f"- **Schema version**: {profile.schema_version}")
        built_at = (
            profile.profile_built_at.strftime("%Y-%m-%d %H:%M UTC")
            if profile.profile_built_at
            else "N/A"
        )
        _add(f"- **Profile built**: {built_at}")
        _add(f"- **Tweets analysed**: {profile.tweet_count_analyzed}")
        date_start = (
            profile.date_range_start.strftime("%Y-%m-%d")
            if profile.date_range_start
            else "N/A"
        )
        date_end = (
            profile.date_range_end.strftime("%Y-%m-%d")
            if profile.date_range_end
            else "N/A"
        )
        _add(f"- **Date range**: {date_start} to {date_end}")
        _add()

        # -- Structural Style --
        ss = profile.structural_style
        _add("## Structural Style")
        _add()
        _add(f"- **Avg tweet length**: {ss.avg_tweet_length:.1f} chars")
        _add(f"- **Avg sentence length**: {ss.avg_sentence_length:.1f} tokens")
        _add(f"- **Fragment ratio**: {ss.fragment_ratio:.1%}")
        _add(f"- **Question ratio**: {ss.question_ratio:.1%}")
        _add(f"- **Exclamation ratio**: {ss.exclamation_ratio:.1%}")
        if ss.vocabulary_richness:
            _add(f"- **Vocabulary richness**: {json.dumps(ss.vocabulary_richness, indent=2)}")
        if ss.top_vocabulary:
            top_words = ", ".join(f"{w} ({c})" for w, c in ss.top_vocabulary[:20])
            _add(f"- **Top vocabulary**: {top_words}")
        _add()

        # -- Readability --
        rs = ss.readability_scores
        _add("## Readability")
        _add()
        _add(f"- **Flesch Reading Ease**: {rs.flesch_reading_ease:.1f}")
        _add(f"- **Coleman-Liau Index**: {rs.coleman_liau_index:.1f}")
        _add(f"- **ARI**: {rs.automated_readability_index:.1f}")
        _add(f"- **Gunning Fog**: {rs.gunning_fog:.1f}")
        _add()

        # -- Emoji --
        ep = profile.emoji_profile
        _add("## Emoji Profile")
        _add()
        _add(f"- **Emojis per tweet**: {ep.usage_rate:.2f}")
        _add(f"- **Tweets with emojis**: {ep.emoji_tweet_ratio:.1%}")
        _add(f"- **Emoji diversity**: {ep.emoji_diversity:.2f}")
        _add(f"- **Emoji sentiment**: {ep.emoji_sentiment:+.2f}")
        if ep.top_emojis:
            top_e = ", ".join(f"{e} ({c})" for e, c in ep.top_emojis[:10])
            _add(f"- **Top emojis**: {top_e}")
        _add()

        # -- Hashtags --
        hp = profile.hashtag_profile
        _add("## Hashtag Profile")
        _add()
        _add(f"- **Hashtags per tweet**: {hp.usage_rate:.2f}")
        _add(f"- **Placement style**: {hp.placement_style}")
        _add(f"- **Casing style**: {hp.casing_style}")
        if hp.top_hashtags:
            top_h = ", ".join(f"#{h} ({c})" for h, c in hp.top_hashtags[:10])
            _add(f"- **Top hashtags**: {top_h}")
        _add()

        # -- Personality --
        pp = profile.personality_profile
        _add("## Personality Profile")
        _add()
        _add(f"- **Formality**: {pp.formality_score:.2f}")
        _add(f"- **Humor style**: {pp.humor_style}")
        _add(f"- **Tone**: {pp.tone}")
        _add(f"- **Assertiveness**: {pp.assertiveness:.2f}")
        b5 = pp.big_five
        _add(f"- **Big Five**: O={b5.openness:.2f} C={b5.conscientiousness:.2f} "
             f"E={b5.extraversion:.2f} A={b5.agreeableness:.2f} N={b5.neuroticism:.2f}")
        if pp.rhetorical_devices:
            _add(f"- **Rhetorical devices**: {', '.join(pp.rhetorical_devices)}")
        if pp.catchphrases:
            _add(f"- **Catchphrases**: {', '.join(repr(c) for c in pp.catchphrases)}")
        _add()

        # -- Reply Style --
        rp = profile.reply_style
        _add("## Reply Style")
        _add()
        _add(f"- **Avg reply length**: {rp.avg_reply_length:.1f} chars")
        _add(f"- **Length ratio** (reply/original): {rp.length_ratio:.2f}")
        _add(f"- **Tone vs originals**: {rp.tone_vs_originals}")
        _add(f"- **Engagement type**: {rp.engagement_type}")
        _add(f"- **Sentiment diff**: {rp.sentiment_diff:+.3f}")
        if rp.opening_patterns:
            _add(f"- **Opening patterns**: {json.dumps(rp.opening_patterns, indent=2)}")
        _add()

        # -- Topics --
        tp = profile.topic_profile
        if tp.topics_ranked:
            _add("## Topics")
            _add()
            for t in tp.topics_ranked[:15]:
                keywords = ", ".join(t.keywords[:5]) if t.keywords else ""
                _add(f"- **{t.name}** (importance: {t.importance:.2f}): {keywords}")
            _add()

        # -- Example Tweets --
        ex = profile.example_tweets
        if ex.originals:
            _add("## Example Original Tweets")
            _add()
            for i, tweet in enumerate(ex.originals[:10], 1):
                _add(f"{i}. {tweet}")
            if len(ex.originals) > 10:
                _add(f"   ... and {len(ex.originals) - 10} more")
            _add()

        if ex.replies:
            _add("## Example Replies")
            _add()
            for i, (context, reply) in enumerate(ex.replies[:5], 1):
                _add(f"{i}. **Context**: {context}")
                _add(f"   **Reply**: {reply}")
                _add()

        return "\n".join(lines)
