"""Data models for raw and cleaned tweets, and user profiles."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RawTweet(BaseModel):
    """A tweet as collected from the X/Twitter API or archive, before any processing.

    This model preserves all original metadata needed for downstream analysis,
    including engagement metrics, reply context, and media information.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique tweet identifier")
    text: str = Field(description="Full tweet text content")
    created_at: datetime = Field(description="UTC timestamp when the tweet was posted")
    is_reply: bool = Field(default=False, description="Whether this tweet is a reply to another")
    reply_to_user: str | None = Field(
        default=None,
        description="Username of the account being replied to, if this is a reply",
    )
    reply_to_tweet_id: str | None = Field(
        default=None,
        description="Tweet ID being replied to, if this is a reply",
    )
    is_retweet: bool = Field(default=False, description="Whether this is a native retweet")
    is_quote_tweet: bool = Field(
        default=False, description="Whether this is a quote tweet with added commentary"
    )
    quoted_text: str | None = Field(
        default=None, description="Text of the quoted tweet, if this is a quote tweet"
    )
    hashtags: list[str] = Field(
        default_factory=list, description="Hashtags used in the tweet (without # prefix)"
    )
    mentions: list[str] = Field(
        default_factory=list, description="Usernames mentioned in the tweet (without @ prefix)"
    )
    urls: list[str] = Field(
        default_factory=list, description="URLs included in the tweet (expanded form)"
    )
    media_types: list[str] = Field(
        default_factory=list,
        description="Types of attached media: 'image', 'video', 'gif'",
    )
    like_count: int = Field(default=0, ge=0, description="Number of likes")
    retweet_count: int = Field(default=0, ge=0, description="Number of retweets")
    reply_count: int = Field(default=0, ge=0, description="Number of replies received")
    language: str | None = Field(
        default=None, description="BCP-47 language code detected by Twitter, e.g. 'en'"
    )


class EmojiPosition(BaseModel):
    """Position and identity of a single emoji occurrence within a tweet."""

    model_config = ConfigDict(frozen=True)

    emoji: str = Field(description="The emoji character(s)")
    match_start: int = Field(ge=0, description="Start character index in the original text")
    match_end: int = Field(ge=0, description="End character index in the original text")
    position_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized position within the tweet (0.0 = start, 1.0 = end)",
    )


class CleanTweet(BaseModel):
    """A tweet after preprocessing: text cleaned, emojis extracted, mentions normalized.

    This is the primary unit of analysis throughout the pipeline. The original
    tweet ID is preserved so we can trace back to the source.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Original tweet identifier for traceability")
    text: str = Field(description="Cleaned tweet text (URLs removed, unicode normalized)")
    original_text: str = Field(description="Original unmodified tweet text")
    created_at: datetime = Field(description="UTC timestamp of the original tweet")
    is_reply: bool = Field(default=False, description="Whether this is a reply")
    reply_to_user: str | None = Field(default=None, description="Username being replied to")
    reply_to_tweet_id: str | None = Field(default=None, description="Tweet ID being replied to")
    emoji_positions: list[EmojiPosition] = Field(
        default_factory=list,
        description="All emojis extracted from the text with their positions",
    )
    original_mentions: list[str] = Field(
        default_factory=list,
        description="Original @usernames before normalization to @USER",
    )
    hashtags: list[str] = Field(default_factory=list, description="Hashtags from the tweet")
    like_count: int = Field(default=0, ge=0, description="Engagement: like count")
    retweet_count: int = Field(default=0, ge=0, description="Engagement: retweet count")
    reply_count: int = Field(default=0, ge=0, description="Engagement: reply count")
    language: str | None = Field(default=None, description="Detected language")
    is_quote_tweet: bool = Field(default=False, description="Whether this is a quote tweet")
    quoted_text: str | None = Field(default=None, description="Text of the quoted tweet")


class UserProfile(BaseModel):
    """Basic public profile information for an X/Twitter account."""

    model_config = ConfigDict(frozen=True)

    username: str = Field(description="Account handle without @ prefix, e.g. 'elonmusk'")
    display_name: str = Field(description="Public display name")
    bio: str = Field(default="", description="Account bio / description")
    followers_count: int = Field(default=0, ge=0, description="Number of followers")
    following_count: int = Field(default=0, ge=0, description="Number of accounts followed")
    tweet_count: int = Field(default=0, ge=0, description="Total number of tweets posted")
