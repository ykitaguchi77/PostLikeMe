"""Reusable tweet card component for Streamlit."""

from __future__ import annotations

import streamlit as st


def render_tweet_card(
    text: str,
    username: str = "",
    score: float | None = None,
    index: int | None = None,
) -> None:
    """Render a tweet in a Twitter-like card.

    Parameters
    ----------
    text:
        The tweet text to display.
    username:
        The @username to show (optional).
    score:
        Style adherence score 0.0-1.0 (optional).
    index:
        Candidate number for labeling (optional).
    """
    score_html = ""
    if score is not None:
        pct = int(score * 100)
        color = "#1da1f2" if pct >= 70 else "#ffad1f" if pct >= 40 else "#e0245e"
        score_html = (
            f'<span class="score-badge" style="background:{color}">'
            f"スタイル一致度: {pct}%</span>"
        )

    username_html = ""
    if username:
        username_html = f'<div class="username">@{username}</div>'

    label = f"候補 {index}" if index is not None else ""

    st.markdown(
        f"""
        <div class="tweet-card">
            {username_html}
            <div>{text}</div>
            {score_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Copy button (Streamlit native)
    if index is not None:
        st.code(text, language=None)


def render_reply_card(
    original_tweet: str,
    reply_text: str,
    username: str = "",
    score: float | None = None,
    index: int | None = None,
) -> None:
    """Render a reply with the original tweet context.

    Parameters
    ----------
    original_tweet:
        The tweet being replied to.
    reply_text:
        The generated reply.
    username:
        The @username replying.
    score:
        Style adherence score.
    index:
        Candidate number.
    """
    score_html = ""
    if score is not None:
        pct = int(score * 100)
        color = "#1da1f2" if pct >= 70 else "#ffad1f" if pct >= 40 else "#e0245e"
        score_html = (
            f'<span class="score-badge" style="background:{color}">'
            f"スタイル一致度: {pct}%</span>"
        )

    st.markdown(
        f"""
        <div class="tweet-card" style="opacity: 0.7; border-left: 3px solid #38444d;">
            <div style="font-size: 13px; color: #8899a6;">元のツイート:</div>
            <div>{original_tweet}</div>
        </div>
        <div class="tweet-card" style="margin-top: -4px; border-left: 3px solid #1da1f2;">
            <div class="username">@{username} のリプライ</div>
            <div>{reply_text}</div>
            {score_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if index is not None:
        st.code(reply_text, language=None)


def render_profile_metric(label: str, value: str | int | float, delta: str = "") -> None:
    """Render a styled metric card."""
    st.metric(label=label, value=value, delta=delta if delta else None)
