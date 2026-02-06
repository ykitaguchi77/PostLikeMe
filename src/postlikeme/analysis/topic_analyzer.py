"""Topic analysis of tweet corpora.

Discovers topics using BERTopic (primary) or NMF + TF-IDF (fallback),
applies recency weighting, and computes per-topic sentiment via VADER.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from postlikeme.models.raw_tweet import CleanTweet
from postlikeme.models.voice_profile import (
    OpinionEntry,
    SemanticCluster,
    TopicEntry,
    TopicProfile,
)
from postlikeme.utils.constants import DEFAULT_HALF_LIFE_DAYS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recency_weight(
    tweet_date: datetime,
    now: datetime,
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Exponential decay weight based on tweet age.

    weight = exp(-lambda * days_ago)
    where lambda = ln(2) / half_life_days
    """
    lambda_ = math.log(2) / half_life_days
    days_ago = max((now - tweet_date).total_seconds() / 86400, 0.0)
    return math.exp(-lambda_ * days_ago)


def _sentiment_label(mean_compound: float) -> str:
    """Classify a mean VADER compound score into a qualitative label."""
    if mean_compound > 0.3:
        return "enthusiastic"
    if mean_compound > 0.1:
        return "generally_positive"
    if mean_compound > -0.1:
        return "neutral"
    if mean_compound > -0.3:
        return "critical"
    return "negative"


def _compute_tweet_sentiments(texts: list[str]) -> list[float]:
    """Compute VADER compound sentiment for each text."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        return [analyzer.polarity_scores(t)["compound"] for t in texts]
    except ImportError:
        logger.warning("vaderSentiment not installed; sentiment scores default to 0.")
        return [0.0] * len(texts)


# ---------------------------------------------------------------------------
# BERTopic implementation
# ---------------------------------------------------------------------------


def _try_bertopic(
    texts: list[str],
) -> tuple[list[int], Any, Any] | None:
    """Attempt to run BERTopic.  Returns (topics, model, embeddings) or None."""
    try:
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN
        from bertopic import BERTopic

        logger.info("Using BERTopic for topic modelling.")

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        umap_model = UMAP(
            n_neighbors=min(10, len(texts) - 1) if len(texts) > 1 else 2,
            n_components=min(5, len(texts) - 1) if len(texts) > 5 else 2,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        min_cluster = min(10, max(2, len(texts) // 20))
        min_samp = min(5, min_cluster)
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster,
            metric="euclidean",
            cluster_selection_method="leaf",
            prediction_data=True,
            min_samples=min_samp,
        )

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            min_topic_size=min_cluster,
            verbose=False,
        )

        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        topics, _probs = topic_model.fit_transform(texts, embeddings=embeddings)

        return topics, topic_model, embeddings

    except ImportError:
        logger.info(
            "sentence-transformers / bertopic not available; falling back to NMF."
        )
        return None
    except Exception as exc:
        logger.warning("BERTopic failed (%s); falling back to NMF.", exc)
        return None


# ---------------------------------------------------------------------------
# NMF fallback implementation
# ---------------------------------------------------------------------------


def _nmf_topics(
    texts: list[str], max_k: int = 30
) -> tuple[list[int], list[list[str]], np.ndarray]:
    """Discover topics using NMF + TF-IDF.

    Returns (topic_assignments, topic_keywords_list, W_matrix).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    # Guard for very small corpora
    min_df = min(3, max(1, len(texts) // 10))
    n_features = min(5000, len(texts) * 10)

    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=min_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=n_features,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Search for best K using reconstruction error
    max_k = min(max_k, max(5, len(texts) // 10))
    min_k = min(5, max_k)

    best_k = min_k
    best_error = float("inf")

    for k in range(min_k, max_k + 1):
        try:
            nmf = NMF(n_components=k, random_state=42, max_iter=400)
            W = nmf.fit_transform(tfidf_matrix)
            error = nmf.reconstruction_err_
            if error < best_error:
                best_error = error
                best_k = k
        except Exception:
            continue

    # Fit with best K
    nmf = NMF(n_components=best_k, random_state=42, max_iter=400)
    W = nmf.fit_transform(tfidf_matrix)

    # Assign each document to its dominant topic
    topic_assignments = W.argmax(axis=1).tolist()

    # Extract top keywords per topic
    topic_keywords: list[list[str]] = []
    for topic_idx in range(best_k):
        top_indices = nmf.components_[topic_idx].argsort()[::-1][:10]
        keywords = [str(feature_names[i]) for i in top_indices]
        topic_keywords.append(keywords)

    return topic_assignments, topic_keywords, W


# ---------------------------------------------------------------------------
# TopicAnalyzer
# ---------------------------------------------------------------------------


class TopicAnalyzer:
    """Discover topics in a tweet corpus, apply recency weighting, and
    compute per-topic sentiment."""

    def analyze(self, tweets: list[CleanTweet]) -> TopicProfile:
        """Run topic analysis and return a ``TopicProfile``."""
        if not tweets:
            logger.warning("No tweets provided to TopicAnalyzer.")
            return TopicProfile()

        if len(tweets) < 10:
            logger.warning(
                "Only %d tweets for topic analysis; results may be unreliable.",
                len(tweets),
            )

        texts = [t.text for t in tweets]
        now = datetime.now(timezone.utc)

        # ------------------------------------------------------------------
        # Step 1: Topic discovery (BERTopic or NMF fallback)
        # ------------------------------------------------------------------
        bertopic_result = _try_bertopic(texts)

        semantic_clusters: list[SemanticCluster] = []
        topic_assignments: list[int]
        topic_keywords_map: dict[int, list[str]] = {}
        topic_names: dict[int, str] = {}

        if bertopic_result is not None:
            topics_raw, topic_model, _embeddings = bertopic_result
            topic_assignments = list(topics_raw)

            # Extract topic info from BERTopic model
            try:
                topic_info = topic_model.get_topic_info()
                for _, row in topic_info.iterrows():
                    tid = int(row["Topic"])
                    if tid == -1:
                        continue  # skip outlier topic
                    topic_words = topic_model.get_topic(tid)
                    keywords = [w for w, _ in topic_words[:10]] if topic_words else []
                    topic_keywords_map[tid] = keywords
                    topic_names[tid] = ", ".join(keywords[:3]) if keywords else f"topic_{tid}"

                    # Build semantic clusters
                    cluster_tweets = [
                        texts[i]
                        for i, t in enumerate(topic_assignments)
                        if t == tid
                    ]
                    semantic_clusters.append(
                        SemanticCluster(
                            cluster_id=tid,
                            label=topic_names.get(tid, ""),
                            size=len(cluster_tweets),
                            keywords=keywords[:10],
                            representative_tweets=cluster_tweets[:5],
                        )
                    )
            except Exception as exc:
                logger.warning("Error extracting BERTopic info: %s", exc)
                # Fall through with raw assignments
                unique_topics = set(t for t in topic_assignments if t != -1)
                for tid in unique_topics:
                    topic_keywords_map[tid] = []
                    topic_names[tid] = f"topic_{tid}"
        else:
            # NMF fallback
            logger.info("Using NMF fallback for topic modelling.")
            topic_assignments_raw, keywords_list, _W = _nmf_topics(texts)
            topic_assignments = topic_assignments_raw

            for tid, kws in enumerate(keywords_list):
                topic_keywords_map[tid] = kws
                topic_names[tid] = ", ".join(kws[:3]) if kws else f"topic_{tid}"

                cluster_tweets = [
                    texts[i]
                    for i, t in enumerate(topic_assignments)
                    if t == tid
                ]
                if cluster_tweets:
                    semantic_clusters.append(
                        SemanticCluster(
                            cluster_id=tid,
                            label=topic_names[tid],
                            size=len(cluster_tweets),
                            keywords=kws[:10],
                            representative_tweets=cluster_tweets[:5],
                        )
                    )

        # ------------------------------------------------------------------
        # Step 2: Recency-weighted topic importance
        # ------------------------------------------------------------------
        topic_importance: dict[int, float] = defaultdict(float)
        for i, tweet in enumerate(tweets):
            tid = topic_assignments[i] if i < len(topic_assignments) else -1
            if tid == -1:
                continue
            # Ensure tweet created_at is timezone-aware
            tweet_dt = tweet.created_at
            if tweet_dt.tzinfo is None:
                tweet_dt = tweet_dt.replace(tzinfo=timezone.utc)
            weight = _recency_weight(tweet_dt, now)
            topic_importance[tid] += weight

        total_importance = sum(topic_importance.values())
        if total_importance > 0:
            topic_importance = {
                k: v / total_importance for k, v in topic_importance.items()
            }

        # Sort topics by importance
        sorted_topics = sorted(topic_importance.items(), key=lambda x: x[1], reverse=True)

        topics_ranked: list[TopicEntry] = []
        for tid, importance in sorted_topics:
            topics_ranked.append(
                TopicEntry(
                    name=topic_names.get(tid, f"topic_{tid}"),
                    importance=round(min(importance, 1.0), 4),
                    keywords=topic_keywords_map.get(tid, []),
                )
            )

        # ------------------------------------------------------------------
        # Step 3: Per-topic sentiment via VADER
        # ------------------------------------------------------------------
        sentiments = _compute_tweet_sentiments(texts)

        # Group sentiments by topic
        topic_sentiments: dict[int, list[float]] = defaultdict(list)
        for i, s in enumerate(sentiments):
            tid = topic_assignments[i] if i < len(topic_assignments) else -1
            if tid != -1:
                topic_sentiments[tid].append(s)

        opinion_map: list[OpinionEntry] = []
        for tid, importance in sorted_topics:
            sents = topic_sentiments.get(tid, [])
            if sents:
                mean_s = float(np.mean(sents))
                std_s = float(np.std(sents))
            else:
                mean_s = 0.0
                std_s = 0.0

            opinion_map.append(
                OpinionEntry(
                    topic=topic_names.get(tid, f"topic_{tid}"),
                    sentiment_mean=round(mean_s, 4),
                    sentiment_std=round(std_s, 4),
                    label=_sentiment_label(mean_s),
                )
            )

        return TopicProfile(
            topics_ranked=topics_ranked,
            opinion_map=opinion_map,
            semantic_clusters=semantic_clusters,
        )
