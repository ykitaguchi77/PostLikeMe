"""FastAPI application for the PostLikeMe REST API.

Exposes endpoints for tweet collection, analysis, generation, and profile
management. Designed for use with a frontend or programmatic access.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from postlikeme.api.routers import analyze, collect, generate, profiles

app = FastAPI(
    title="PostLikeMe API",
    version="0.1.0",
    description="Analyze X/Twitter accounts and generate style-matched tweets.",
)

# CORS middleware -- allow all origins for development; restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(collect.router, prefix="/api", tags=["collect"])
app.include_router(analyze.router, prefix="/api", tags=["analyze"])
app.include_router(generate.router, prefix="/api", tags=["generate"])
app.include_router(profiles.router, prefix="/api", tags=["profiles"])


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
