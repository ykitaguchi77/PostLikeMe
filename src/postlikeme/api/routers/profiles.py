"""API router for voice profile management."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from postlikeme.api.schemas import ProfileSummary
from postlikeme.models.config import AppConfig
from postlikeme.models.voice_profile import VoiceProfile
from postlikeme.storage.config_store import ConfigStore
from postlikeme.storage.profile_store import ProfileStore

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_profile_store() -> ProfileStore:
    """Create and return a ProfileStore from current configuration."""
    config_store = ConfigStore(AppConfig().data_dir)
    config = config_store.load()
    config.ensure_dirs()
    return ProfileStore(config.data_dir)


@router.get("/profiles", response_model=list[ProfileSummary])
async def list_profiles() -> list[ProfileSummary]:
    """List all stored voice profiles.

    Returns a summary of each profile including username, display name,
    build timestamp, and tweet count.
    """
    profile_store = _get_profile_store()
    usernames = profile_store.list_profiles()

    summaries: list[ProfileSummary] = []
    for uname in usernames:
        profile = profile_store.load_profile(uname)
        if profile is None:
            continue
        summaries.append(
            ProfileSummary(
                username=profile.username,
                display_name=profile.display_name,
                built_at=profile.profile_built_at,
                tweet_count_analyzed=profile.tweet_count_analyzed,
            )
        )

    return summaries


@router.get("/profiles/{username}")
async def get_profile(username: str) -> dict:
    """Retrieve the full voice profile for a user.

    Returns the complete profile as a JSON object, including all
    analysis sections (structural style, topics, personality, etc.).
    """
    username = username.lstrip("@").lower()
    profile_store = _get_profile_store()

    profile = profile_store.load_profile(username)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for @{username}. "
            f"Run collection and analysis first.",
        )

    # Return the full profile as a dict (Pydantic model serializes all fields)
    return profile.model_dump(mode="json")


@router.delete("/profiles/{username}")
async def delete_profile(username: str) -> dict[str, str]:
    """Delete a stored voice profile.

    Permanently removes the profile JSON file from disk.
    """
    username = username.lstrip("@").lower()
    profile_store = _get_profile_store()

    # Check if profile exists
    profile = profile_store.load_profile(username)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for @{username}.",
        )

    deleted = profile_store.delete_profile(username)
    if not deleted:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete profile for @{username}.",
        )

    return {"message": f"Profile for @{username} has been deleted."}
