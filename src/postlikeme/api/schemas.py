"""Pydantic schemas for the PostLikeMe REST API.

These models define the request and response shapes for all API endpoints.
They are intentionally separate from the internal domain models to allow
the API contract to evolve independently.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class CollectRequest(BaseModel):
    """Request body for the tweet collection endpoint."""

    username: str = Field(description="X/Twitter username to collect tweets from.")
    count: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum number of tweets to collect.",
    )
    collector: str = Field(
        default="api",
        description="Collection backend: 'api', 'scrape', or 'archive'.",
    )
    include_replies: bool = Field(
        default=True,
        description="Whether to include reply tweets.",
    )


class CollectResponse(BaseModel):
    """Response body after successful tweet collection."""

    username: str = Field(description="Username that was collected.")
    tweet_count: int = Field(description="Number of tweets collected.")
    message: str = Field(description="Human-readable status message.")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Request body for the analysis endpoint."""

    username: str = Field(description="Username to analyse (must already be collected).")
    rebuild: bool = Field(
        default=False,
        description="Force rebuild even if a profile already exists.",
    )


class AnalyzeResponse(BaseModel):
    """Response body after successful analysis."""

    username: str = Field(description="Username that was analysed.")
    tweet_count_analyzed: int = Field(description="Number of tweets used in analysis.")
    topics_found: int = Field(description="Number of topics discovered.")
    message: str = Field(description="Human-readable status message.")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class GenerateTweetRequest(BaseModel):
    """Request body for tweet generation."""

    username: str = Field(description="Username whose style to emulate.")
    topic: Optional[str] = Field(
        default=None,
        description="Optional topic to guide the generation.",
    )
    count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of tweet candidates to generate.",
    )
    temperature: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature.",
    )


class GenerateReplyRequest(BaseModel):
    """Request body for reply generation."""

    username: str = Field(description="Username whose style to emulate.")
    target_tweet: str = Field(description="The tweet text to generate a reply to.")
    count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of reply candidates to generate.",
    )
    temperature: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature.",
    )


class GenerateCandidate(BaseModel):
    """A single generated tweet/reply candidate with its quality score."""

    text: str = Field(description="The generated text.")
    score: float = Field(description="Style adherence score (0.0 to 1.0).")


class GenerateResponse(BaseModel):
    """Response body for tweet/reply generation."""

    candidates: list[GenerateCandidate] = Field(
        default_factory=list,
        description="Generated candidates ranked by style adherence.",
    )


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class ProfileSummary(BaseModel):
    """Summary of a stored voice profile."""

    username: str = Field(description="Account handle.")
    display_name: str = Field(default="", description="Public display name.")
    built_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the profile was built.",
    )
    tweet_count_analyzed: int = Field(
        default=0,
        description="Number of tweets used to build the profile.",
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(description="Human-readable error message.")
