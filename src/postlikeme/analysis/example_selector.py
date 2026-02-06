"""Example tweet selection using engagement scoring and MMR diversity.

Selects a representative set of original tweets and reply pairs for
few-shot prompting in the generation engine. Uses engagement-weighted
scoring combined with Maximal Marginal Relevance (MMR) to ensure both
quality and diversity in the selected examples.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np

from postlikeme.models.raw_tweet import CleanTweet
from postlikeme.models.voice_profile import ExampleTweets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engagement scoring
# ---------------------------------------------------------------------------


def _engagement_score(tweet: CleanTweet) -> float:
    """Compute a composite engagement score for a tweet.

    score = log(1 + likes) + 1.5 * log(1 + retweets) + 0.5 * log(1 + replies)

    This weights retweets most heavily (viral signal), then likes
    (approval), then replies (conversation).
    """
    return (
        math.log(1 + tweet.like_count)
        + 1.5 * math.log(1 + tweet.retweet_count)
        + 0.5 * math.log(1 + tweet.reply_count)
    )


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (lightweight, no sentence-transformers needed)
# ---------------------------------------------------------------------------


def _build_tfidf_matrix(texts: list[str]) -> np.ndarray:
    """Build a TF-IDF matrix from a list of texts.

    Uses scikit-learn's TfidfVectorizer if available; otherwise falls back
    to a simple term-frequency approach with IDF weighting.

    Returns
    -------
    np.ndarray
        Shape (n_texts, n_features) TF-IDF matrix (sparse converted to dense).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_df=0.9,
            min_df=1,
        )
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray()
    except ImportError:
        logger.warning(
            "scikit-learn not available; using basic term-frequency for diversity selection."
        )
        return _basic_tf_matrix(texts)


def _basic_tf_matrix(texts: list[str]) -> np.ndarray:
    """Fallback TF matrix without scikit-learn.

    Builds a simple bag-of-words matrix with log(1+count) weighting
    and L2 normalization.
    """
    # Build vocabulary from all texts
    vocab: dict[str, int] = {}
    tokenized: list[list[str]] = []
    for text in texts:
        tokens = text.lower().split()
        tokenized.append(tokens)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    if not vocab:
        return np.zeros((len(texts), 1))

    # Build count matrix
    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for i, tokens in enumerate(tokenized):
        for token in tokens:
            matrix[i, vocab[token]] += 1.0

    # Apply log(1+count) weighting
    matrix = np.log1p(matrix)

    # IDF weighting
    doc_freq = np.sum(matrix > 0, axis=0) + 1  # add 1 to avoid division by zero
    idf = np.log(len(texts) / doc_freq)
    matrix *= idf

    # L2 normalize each row
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix /= norms

    return matrix


def _cosine_similarity_matrix(tfidf: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    The input is already L2-normalized (or close to it for sklearn),
    so cosine similarity is just the dot product.

    Returns
    -------
    np.ndarray
        Shape (n, n) similarity matrix with values in [-1, 1].
    """
    # L2 normalize rows to be safe
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = tfidf / norms
    return normalized @ normalized.T


# ---------------------------------------------------------------------------
# MMR selection
# ---------------------------------------------------------------------------


def _mmr_select(
    candidates: list[int],
    scores: np.ndarray,
    sim_matrix: np.ndarray,
    target: int,
    lambda_param: float = 0.4,
) -> list[int]:
    """Select indices using Maximal Marginal Relevance.

    MMR balances relevance (engagement score) against diversity
    (dissimilarity to already-selected items).

    MMR(i) = lambda * score(i) - (1 - lambda) * max_{j in selected} sim(i, j)

    Parameters
    ----------
    candidates:
        List of candidate indices to choose from.
    scores:
        Relevance scores for each candidate (higher = better).
    sim_matrix:
        Pairwise similarity matrix (n x n).
    target:
        Number of items to select.
    lambda_param:
        Trade-off parameter. Lower values favour diversity over relevance.

    Returns
    -------
    list[int]
        Selected indices in order of selection.
    """
    if len(candidates) <= target:
        return list(candidates)

    # Normalize scores to [0, 1]
    score_arr = np.array([scores[i] for i in candidates])
    max_score = score_arr.max()
    min_score = score_arr.min()
    if max_score > min_score:
        norm_scores = (score_arr - min_score) / (max_score - min_score)
    else:
        norm_scores = np.ones_like(score_arr)

    selected: list[int] = []
    remaining = set(range(len(candidates)))

    for _ in range(target):
        if not remaining:
            break

        best_mmr = -float("inf")
        best_idx = -1

        for r_idx in remaining:
            cand_idx = candidates[r_idx]
            relevance = norm_scores[r_idx]

            # Max similarity to already-selected items
            if selected:
                sel_indices = [candidates[s] for s in selected]
                max_sim = max(sim_matrix[cand_idx, s] for s in sel_indices)
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = r_idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.discard(best_idx)

    return [candidates[s] for s in selected]


# ---------------------------------------------------------------------------
# ExampleSelector
# ---------------------------------------------------------------------------


class ExampleSelector:
    """Select diverse, high-quality example tweets for few-shot prompting.

    Uses engagement-weighted scoring combined with MMR diversity selection
    to produce a representative sample of both original tweets and reply pairs.
    """

    def select(
        self,
        originals: list[CleanTweet],
        replies: list[CleanTweet],
        topics: list[str] | None = None,
        target_originals: int = 25,
        target_replies: int = 15,
    ) -> ExampleTweets:
        """Select representative examples from the cleaned tweet corpus.

        Parameters
        ----------
        originals:
            Cleaned original (non-reply) tweets.
        replies:
            Cleaned reply tweets.
        topics:
            Optional list of topic labels (not used in current selection
            but reserved for topic-balanced selection in future).
        target_originals:
            Desired number of original tweet examples.
        target_replies:
            Desired number of reply pair examples.

        Returns
        -------
        ExampleTweets
            Curated set of original tweets and (context, reply) pairs.
        """
        selected_originals = self._select_originals(originals, target_originals)
        selected_replies = self._select_replies(replies, target_replies)

        logger.info(
            "Selected %d original examples and %d reply examples.",
            len(selected_originals),
            len(selected_replies),
        )

        return ExampleTweets(
            originals=selected_originals,
            replies=selected_replies,
        )

    def _select_originals(
        self,
        originals: list[CleanTweet],
        target: int,
    ) -> list[str]:
        """Select diverse, high-engagement original tweets via MMR."""
        if not originals:
            return []

        # If fewer than target, return all
        if len(originals) <= target:
            return [t.original_text for t in originals]

        # Filter out very short tweets (< 10 chars) which are not useful as examples
        viable = [t for t in originals if len(t.text.strip()) >= 10]
        if not viable:
            return [t.original_text for t in originals[:target]]

        if len(viable) <= target:
            return [t.original_text for t in viable]

        # Compute engagement scores
        engagement = np.array([_engagement_score(t) for t in viable])

        # Build TF-IDF similarity matrix
        texts = [t.text for t in viable]
        tfidf = _build_tfidf_matrix(texts)
        sim_matrix = _cosine_similarity_matrix(tfidf)

        # Run MMR selection
        candidate_indices = list(range(len(viable)))
        selected_indices = _mmr_select(
            candidate_indices,
            engagement,
            sim_matrix,
            target=target,
            lambda_param=0.4,
        )

        return [viable[i].original_text for i in selected_indices]

    def _select_replies(
        self,
        replies: list[CleanTweet],
        target: int,
    ) -> list[tuple[str, str]]:
        """Select diverse reply (context, reply) pairs via MMR."""
        if not replies:
            return []

        # If fewer than target, return all
        if len(replies) <= target:
            return [self._make_reply_pair(r) for r in replies]

        # Filter out very short replies
        viable = [r for r in replies if len(r.text.strip()) >= 10]
        if not viable:
            return [self._make_reply_pair(r) for r in replies[:target]]

        if len(viable) <= target:
            return [self._make_reply_pair(r) for r in viable]

        # Compute engagement scores
        engagement = np.array([_engagement_score(r) for r in viable])

        # Build TF-IDF similarity matrix
        texts = [r.text for r in viable]
        tfidf = _build_tfidf_matrix(texts)
        sim_matrix = _cosine_similarity_matrix(tfidf)

        # Run MMR selection
        candidate_indices = list(range(len(viable)))
        selected_indices = _mmr_select(
            candidate_indices,
            engagement,
            sim_matrix,
            target=target,
            lambda_param=0.4,
        )

        return [self._make_reply_pair(viable[i]) for i in selected_indices]

    @staticmethod
    def _make_reply_pair(reply: CleanTweet) -> tuple[str, str]:
        """Create a (context, reply) pair from a reply tweet.

        Uses the reply_to_user to construct a context placeholder if the
        original tweet text is not available.
        """
        if reply.reply_to_user:
            context = f"@{reply.reply_to_user}'s tweet"
        else:
            context = "[original tweet]"

        # Use the original text (before mention normalization) for the reply
        reply_text = reply.original_text

        return (context, reply_text)
