"""Structural analysis of tweet corpora.

Computes vocabulary richness, sentence structure, readability, punctuation
patterns, capitalization, emoji usage, hashtag patterns, tweet length
distribution, and character-level stylometry features.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from postlikeme.models.raw_tweet import CleanTweet
from postlikeme.models.voice_profile import (
    CapitalizationPatterns,
    EmojiProfile,
    HashtagProfile,
    PunctuationPatterns,
    ReadabilityScores,
    StructuralStyle,
    TweetLengthDistribution,
)
from postlikeme.utils.constants import CONTRACTION_MAP, FUNCTION_WORDS
from postlikeme.utils.text import detect_contractions, is_all_caps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy singleton -- loaded once, reused across calls
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    """Load the spaCy English model lazily."""
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Downloading... Run: python -m spacy download en_core_web_sm"
            )
            from spacy.cli import download

            download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
        # Increase max length for concatenated corpora
        _NLP.max_length = 5_000_000
    return _NLP


# ---------------------------------------------------------------------------
# StructuralAnalyzer
# ---------------------------------------------------------------------------


class StructuralAnalyzer:
    """Compute all structural, stylometric, and surface-level features
    from a list of ``CleanTweet`` objects.

    Returns a ``StructuralStyle`` plus separate ``EmojiProfile`` and
    ``HashtagProfile`` objects (accessible via the ``analyze_full`` method).
    """

    def analyze(self, tweets: list[CleanTweet]) -> StructuralStyle:
        """Run the full structural analysis and return a ``StructuralStyle``.

        This is the primary entry point.  Use ``analyze_full`` if you also
        need the emoji and hashtag profiles.
        """
        structural, _, _ = self.analyze_full(tweets)
        return structural

    def analyze_full(
        self, tweets: list[CleanTweet]
    ) -> tuple[StructuralStyle, EmojiProfile, HashtagProfile]:
        """Run full structural analysis returning structural style, emoji
        profile, and hashtag profile."""

        if not tweets:
            logger.warning("No tweets provided to StructuralAnalyzer.")
            return StructuralStyle(), EmojiProfile(), HashtagProfile()

        nlp = _get_nlp()

        # -- Collect raw texts ------------------------------------------------
        texts = [t.text for t in tweets]
        corpus = " ".join(texts)

        # -- spaCy batch processing -------------------------------------------
        logger.info("Running spaCy pipeline on %d tweets...", len(tweets))
        docs = list(nlp.pipe(texts, batch_size=64, disable=["ner"]))

        # Corpus-level doc for vocabulary analysis
        corpus_doc = nlp(corpus)

        # =====================================================================
        # 1. VOCABULARY ANALYSIS
        # =====================================================================
        vocabulary_richness = self._compute_vocabulary_richness(corpus_doc, corpus)
        top_vocabulary = self._compute_top_vocabulary(corpus_doc)

        # =====================================================================
        # 2. SENTENCE STRUCTURE
        # =====================================================================
        sentence_stats = self._compute_sentence_structure(docs)

        # =====================================================================
        # 3. READABILITY
        # =====================================================================
        readability = self._compute_readability(corpus)

        # =====================================================================
        # 4. PUNCTUATION PATTERNS
        # =====================================================================
        punctuation = self._compute_punctuation(texts, corpus_doc)

        # =====================================================================
        # 5. CAPITALIZATION
        # =====================================================================
        capitalization = self._compute_capitalization(docs)

        # =====================================================================
        # 6. TWEET LENGTH DISTRIBUTION
        # =====================================================================
        tweet_length_dist = self._compute_tweet_length_distribution(texts)

        # =====================================================================
        # 7. CHARACTER STYLOMETRY
        # =====================================================================
        raw_stats = self._compute_character_stylometry(texts, corpus_doc)

        # =====================================================================
        # 8. EMOJI ANALYSIS
        # =====================================================================
        emoji_profile = self._compute_emoji_profile(tweets)

        # =====================================================================
        # 9. HASHTAG ANALYSIS
        # =====================================================================
        hashtag_profile = self._compute_hashtag_profile(tweets)

        # -- Assemble StructuralStyle -----------------------------------------
        structural = StructuralStyle(
            avg_tweet_length=tweet_length_dist.mean,
            tweet_length_distribution=tweet_length_dist,
            avg_sentence_length=sentence_stats["avg_sentence_length"],
            vocabulary_richness=vocabulary_richness,
            top_vocabulary=top_vocabulary,
            punctuation_patterns=punctuation,
            capitalization_patterns=capitalization,
            fragment_ratio=sentence_stats["fragment_ratio"],
            question_ratio=sentence_stats["question_ratio"],
            exclamation_ratio=sentence_stats["exclamation_ratio"],
            pos_distribution=sentence_stats["pos_distribution"],
            readability_scores=readability,
            raw_statistics=raw_stats,  # type: ignore[call-arg]
        )

        return structural, emoji_profile, hashtag_profile

    # =====================================================================
    # VOCABULARY ANALYSIS
    # =====================================================================

    def _compute_vocabulary_richness(
        self, corpus_doc, corpus_text: str
    ) -> dict[str, float]:
        """Compute TTR, Root TTR, CTTR, MTLD, MATTR, Yule's K/I,
        Hapax ratio, Sichel's S, Honore's R, Brunet's W."""
        tokens = [
            t.text.lower()
            for t in corpus_doc
            if t.is_alpha and not t.is_space
        ]
        N = len(tokens)
        if N == 0:
            return {}

        freq = Counter(tokens)
        V = len(freq)  # unique types

        # Basic ratios
        ttr = V / N if N > 0 else 0.0
        root_ttr = V / math.sqrt(N) if N > 0 else 0.0
        cttr = V / math.sqrt(2 * N) if N > 0 else 0.0

        # --- MTLD and MATTR via lexicalrichness ---
        mtld_val = 0.0
        mattr_val = 0.0
        try:
            from lexicalrichness import LexicalRichness

            lr = LexicalRichness(corpus_text)
            try:
                mtld_val = float(lr.mtld(threshold=0.72))
            except Exception:
                mtld_val = 0.0
            try:
                window = min(50, N)
                if window > 0:
                    mattr_val = float(lr.mattr(window_size=window))
            except Exception:
                mattr_val = 0.0
        except ImportError:
            logger.warning("lexicalrichness not installed; MTLD/MATTR will be 0.")

        # --- Yule's K ---
        # freq_spectrum: f_v(i) = number of types occurring exactly i times
        freq_spectrum: dict[int, int] = Counter(freq.values())
        M1 = N
        M2 = sum(i * i * f_v for i, f_v in freq_spectrum.items())
        if M1 > 0 and (M2 - M1) != 0:
            yules_k = 10_000 * (M2 - M1) / (M1 * M1)
        else:
            yules_k = 0.0

        # Yule's I (inverse of K)
        yules_i = (M1 * M1) / (M2 - M1) if (M2 - M1) != 0 else 0.0

        # --- Hapax-based metrics ---
        V1 = freq_spectrum.get(1, 0)  # hapax legomena
        V2 = freq_spectrum.get(2, 0)  # dis legomena
        hapax_ratio = V1 / V if V > 0 else 0.0
        sichels_s = V2 / V if V > 0 else 0.0

        # Honore's R: 100 * log(N) / (1 - V1/V)
        denom = 1 - (V1 / V) if V > 0 else 1
        if denom > 0 and N > 0:
            honores_r = 100 * math.log(N) / denom
        else:
            honores_r = 0.0

        # Brunet's W: N^(V^(-0.172))
        if V > 0 and N > 0:
            brunets_w = N ** (V ** (-0.172))
        else:
            brunets_w = 0.0

        # vocd-D placeholder (computationally expensive, approximate)
        vocd_d = self._estimate_vocd_d(tokens)

        return {
            "ttr": round(ttr, 4),
            "root_ttr": round(root_ttr, 4),
            "cttr": round(cttr, 4),
            "mtld": round(mtld_val, 2),
            "mattr": round(mattr_val, 4),
            "yules_k": round(yules_k, 2),
            "yules_i": round(yules_i, 2),
            "vocd_d": round(vocd_d, 2),
            "hapax_ratio": round(hapax_ratio, 4),
            "sichels_s": round(sichels_s, 4),
            "honores_r": round(honores_r, 2),
            "brunets_w": round(brunets_w, 4),
        }

    def _estimate_vocd_d(self, tokens: list[str]) -> float:
        """Estimate vocd-D using random sub-sampling.

        Fits the curve TTR = (D/N)[sqrt(1 + 2N/D) - 1] over random
        sub-samples of sizes 35-50.
        """
        if len(tokens) < 50:
            return 0.0

        rng = np.random.default_rng(42)
        observed: list[tuple[int, float]] = []

        for size in range(35, 51):
            ttrs: list[float] = []
            for _ in range(100):
                sample = rng.choice(tokens, size=size, replace=False)
                ttrs.append(len(set(sample)) / size)
            observed.append((size, float(np.mean(ttrs))))

        # Fit D by minimizing sum of squared residuals
        best_d = 0.0
        best_err = float("inf")
        for d_candidate in np.linspace(10, 200, 200):
            err = 0.0
            for n, obs_ttr in observed:
                if d_candidate > 0:
                    pred = (d_candidate / n) * (math.sqrt(1 + 2 * n / d_candidate) - 1)
                else:
                    pred = 0.0
                err += (obs_ttr - pred) ** 2
            if err < best_err:
                best_err = err
                best_d = d_candidate

        return float(best_d)

    def _compute_top_vocabulary(self, corpus_doc) -> list[tuple[str, int]]:
        """Top 50 non-stopword lemmatized tokens."""
        counter: Counter = Counter()
        for token in corpus_doc:
            if (
                token.is_alpha
                and not token.is_stop
                and not token.is_space
                and len(token.text) > 1
            ):
                counter[token.lemma_.lower()] += 1
        return counter.most_common(50)

    # =====================================================================
    # SENTENCE STRUCTURE
    # =====================================================================

    def _compute_sentence_structure(self, docs: list) -> dict[str, Any]:
        """Compute sentence-level statistics using spaCy sentence segmenter."""
        all_sentences = []
        sentence_lengths: list[int] = []
        fragment_count = 0
        question_count = 0
        exclamation_count = 0

        # POS counters
        pos_counter: Counter = Counter()
        total_tokens = 0

        for doc in docs:
            for sent in doc.sents:
                all_sentences.append(sent)
                tokens_in_sent = [t for t in sent if not t.is_space]
                sent_len = len(tokens_in_sent)
                sentence_lengths.append(sent_len)

                # Count POS tags
                for token in tokens_in_sent:
                    pos_counter[token.pos_] += 1
                    total_tokens += 1

                # Fragment detection via dependency parse
                root_token = None
                for token in sent:
                    if token.dep_ == "ROOT":
                        root_token = token
                        break

                if root_token is not None:
                    is_fragment = self._is_fragment(root_token)
                    if is_fragment:
                        fragment_count += 1

                # Question / exclamation detection
                sent_text = sent.text.strip()
                if sent_text.endswith("?"):
                    question_count += 1
                elif sent_text.endswith("!"):
                    exclamation_count += 1

        num_sentences = len(all_sentences)
        if num_sentences == 0:
            return {
                "avg_sentence_length": 0.0,
                "std_sentence_length": 0.0,
                "median_sentence_length": 0.0,
                "fragment_ratio": 0.0,
                "question_ratio": 0.0,
                "exclamation_ratio": 0.0,
                "pos_distribution": {},
            }

        lengths_arr = np.array(sentence_lengths, dtype=float)

        # POS distribution ratios
        pos_dist: dict[str, float] = {}
        if total_tokens > 0:
            pos_dist["noun_ratio"] = round(
                (pos_counter.get("NOUN", 0) + pos_counter.get("PROPN", 0)) / total_tokens, 4
            )
            pos_dist["verb_ratio"] = round(
                (pos_counter.get("VERB", 0) + pos_counter.get("AUX", 0)) / total_tokens, 4
            )
            pos_dist["adj_ratio"] = round(pos_counter.get("ADJ", 0) / total_tokens, 4)
            pos_dist["adv_ratio"] = round(pos_counter.get("ADV", 0) / total_tokens, 4)
            pos_dist["pronoun_ratio"] = round(pos_counter.get("PRON", 0) / total_tokens, 4)
            function_pos_count = sum(
                pos_counter.get(p, 0)
                for p in ("DET", "ADP", "CCONJ", "SCONJ", "PART", "PRON", "AUX")
            )
            pos_dist["function_word_ratio"] = round(function_pos_count / total_tokens, 4)

        return {
            "avg_sentence_length": round(float(np.mean(lengths_arr)), 2),
            "std_sentence_length": round(float(np.std(lengths_arr)), 2),
            "median_sentence_length": round(float(np.median(lengths_arr)), 2),
            "fragment_ratio": round(fragment_count / num_sentences, 4),
            "question_ratio": round(question_count / num_sentences, 4),
            "exclamation_ratio": round(exclamation_count / num_sentences, 4),
            "pos_distribution": pos_dist,
        }

    @staticmethod
    def _is_fragment(root_token) -> bool:
        """Determine if a sentence is a fragment based on its ROOT token.

        Fragment = ROOT is NOUN/ADJ without a VERB child, or ROOT is INTJ.
        """
        if root_token.pos_ == "INTJ":
            return True
        if root_token.pos_ in ("NOUN", "ADJ", "PROPN"):
            has_verb_child = any(c.pos_ in ("VERB", "AUX") for c in root_token.children)
            if not has_verb_child:
                return True
        if root_token.pos_ in ("VERB", "AUX"):
            # Check if it has a subject; no subject and not imperative -> fragment
            has_subj = any(c.dep_ in ("nsubj", "nsubjpass", "csubj") for c in root_token.children)
            if not has_subj:
                morph = root_token.morph.get("VerbForm")
                if morph and "Imp" in morph:
                    return False  # imperative, not a fragment
                # verb fragment (no subject, not imperative)
                return True
        return False

    # =====================================================================
    # READABILITY
    # =====================================================================

    def _compute_readability(self, corpus: str) -> ReadabilityScores:
        """Compute readability on the full concatenated corpus using textstat."""
        if not corpus.strip():
            return ReadabilityScores()

        try:
            import textstat

            textstat.set_lang("en")
            return ReadabilityScores(
                flesch_reading_ease=round(textstat.flesch_reading_ease(corpus), 2),
                coleman_liau_index=round(textstat.coleman_liau_index(corpus), 2),
                automated_readability_index=round(
                    textstat.automated_readability_index(corpus), 2
                ),
                gunning_fog=round(textstat.gunning_fog(corpus), 2),
            )
        except ImportError:
            logger.warning("textstat not installed; readability scores will be 0.")
            return ReadabilityScores()

    # =====================================================================
    # PUNCTUATION PATTERNS
    # =====================================================================

    def _compute_punctuation(
        self, texts: list[str], corpus_doc
    ) -> PunctuationPatterns:
        """Count punctuation per 1000 tokens."""
        total_tokens = max(len([t for t in corpus_doc if not t.is_space]), 1)
        total_tweets = max(len(texts), 1)
        full_text = " ".join(texts)

        period_count = full_text.count(".")
        comma_count = full_text.count(",")
        excl_count = full_text.count("!")
        question_count = full_text.count("?")
        ellipsis_count = full_text.count("...") + full_text.count("\u2026")
        em_dash_count = full_text.count("\u2014") + full_text.count("--")
        paren_count = full_text.count("(") + full_text.count(")")
        # Count both smart and straight quotes
        quot_count = (
            full_text.count('"')
            + full_text.count("\u201c")
            + full_text.count("\u201d")
            + full_text.count("'")
            + full_text.count("\u2018")
            + full_text.count("\u2019")
        )
        semi_count = full_text.count(";")
        colon_count = full_text.count(":")

        per_1k = 1000.0 / total_tokens

        return PunctuationPatterns(
            period=round(period_count * per_1k, 2),
            comma=round(comma_count * per_1k, 2),
            exclamation=round(excl_count * per_1k, 2),
            question=round(question_count * per_1k, 2),
            ellipsis=round(ellipsis_count / total_tweets, 4),  # per tweet
            em_dash=round(em_dash_count * per_1k, 2),
            parentheses=round(paren_count * per_1k, 2),
            quotation=round(quot_count * per_1k, 2),
            semicolon=round(semi_count * per_1k, 2),
            colon=round(colon_count * per_1k, 2),
        )

    # =====================================================================
    # CAPITALIZATION
    # =====================================================================

    def _compute_capitalization(self, docs: list) -> CapitalizationPatterns:
        """Compute capitalization pattern ratios."""
        total_tokens = 0
        all_caps_count = 0
        mid_caps_count = 0
        total_sentences = 0
        initial_caps_count = 0
        lowercase_start_count = 0

        for doc in docs:
            for sent in doc.sents:
                total_sentences += 1
                tokens_in_sent = [t for t in sent if not t.is_space and t.is_alpha]
                if not tokens_in_sent:
                    continue

                first_token = tokens_in_sent[0]
                if first_token.text[0].isupper():
                    initial_caps_count += 1
                elif first_token.text[0].islower():
                    lowercase_start_count += 1

                for i, token in enumerate(tokens_in_sent):
                    total_tokens += 1
                    if is_all_caps(token.text):
                        all_caps_count += 1
                    elif i > 0 and token.text[0].isupper() and not token.pos_ == "PROPN":
                        # Mid-sentence capitalization (not proper noun, not first word)
                        mid_caps_count += 1

        if total_tokens == 0:
            return CapitalizationPatterns()

        return CapitalizationPatterns(
            all_caps_ratio=round(all_caps_count / total_tokens, 4),
            initial_caps_ratio=round(
                initial_caps_count / total_sentences if total_sentences else 0.0, 4
            ),
            lowercase_start_ratio=round(
                lowercase_start_count / total_sentences if total_sentences else 0.0, 4
            ),
            mid_caps_ratio=round(mid_caps_count / total_tokens, 4),
        )

    # =====================================================================
    # TWEET LENGTH DISTRIBUTION
    # =====================================================================

    def _compute_tweet_length_distribution(
        self, texts: list[str]
    ) -> TweetLengthDistribution:
        """Compute full tweet length distribution."""
        lengths = [len(t) for t in texts]
        if not lengths:
            return TweetLengthDistribution(
                mean=0.0, median=0.0, std=0.0, min=0, max=0, p25=0.0, p75=0.0
            )

        arr = np.array(lengths, dtype=float)
        bins = [0, 50, 100, 140, 200, 280]
        hist_counts, _ = np.histogram(arr, bins=bins)
        histogram: dict[str, int] = {}
        for i in range(len(bins) - 1):
            label = f"{bins[i]}-{bins[i + 1]}"
            histogram[label] = int(hist_counts[i])

        return TweetLengthDistribution(
            mean=round(float(np.mean(arr)), 2),
            median=round(float(np.median(arr)), 2),
            std=round(float(np.std(arr)), 2),
            min=int(np.min(arr)),
            max=int(np.max(arr)),
            p25=round(float(np.percentile(arr, 25)), 2),
            p75=round(float(np.percentile(arr, 75)), 2),
            histogram=histogram,
        )

    # =====================================================================
    # CHARACTER STYLOMETRY
    # =====================================================================

    def _compute_character_stylometry(
        self, texts: list[str], corpus_doc
    ) -> dict[str, Any]:
        """Compute function word vector, char n-grams, contraction prefs."""
        raw_stats: dict[str, Any] = {}

        # --- Function word frequency vector ---
        total_tokens = max(
            len([t for t in corpus_doc if not t.is_space and t.is_alpha]), 1
        )
        token_counter: Counter = Counter()
        for t in corpus_doc:
            if t.is_alpha and not t.is_space:
                token_counter[t.text.lower()] += 1

        fw_vector: dict[str, float] = {}
        for word in FUNCTION_WORDS:
            fw_vector[word] = round(token_counter.get(word, 0) / total_tokens, 6)
        raw_stats["function_word_vector"] = fw_vector

        # --- Character trigram top-100 ---
        try:
            from sklearn.feature_extraction.text import CountVectorizer

            char_vectorizer = CountVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                max_features=200,
            )
            char_matrix = char_vectorizer.fit_transform(texts)
            feature_names = char_vectorizer.get_feature_names_out()
            total_ngrams = char_matrix.sum()
            if total_ngrams > 0:
                freqs = np.asarray(char_matrix.sum(axis=0)).flatten()
                top_indices = freqs.argsort()[::-1][:100]
                top_trigrams: list[tuple[str, float]] = []
                for idx in top_indices:
                    name = feature_names[idx]
                    norm_freq = float(freqs[idx]) / float(total_ngrams)
                    top_trigrams.append((name, round(norm_freq, 6)))
                raw_stats["char_ngram_top100"] = top_trigrams
            else:
                raw_stats["char_ngram_top100"] = []
        except ImportError:
            logger.warning("scikit-learn not installed; char n-gram analysis skipped.")
            raw_stats["char_ngram_top100"] = []

        # --- Contraction preference ratios ---
        all_text = " ".join(texts)
        all_text_lower = all_text.lower()
        contraction_prefs: dict[str, float] = {}
        for contracted, expanded in CONTRACTION_MAP.items():
            contracted_count = all_text_lower.count(contracted)
            # Search for the expanded form (case-insensitive, word boundary)
            expanded_pattern = r"\b" + re.escape(expanded) + r"\b"
            expanded_count = len(re.findall(expanded_pattern, all_text_lower))
            total = contracted_count + expanded_count
            if total > 0:
                contraction_prefs[contracted] = round(contracted_count / total, 4)
        raw_stats["contraction_preferences"] = contraction_prefs

        # --- Additional metrics used downstream ---
        # Contraction ratio (total contractions / total words)
        all_contractions = detect_contractions(all_text)
        raw_stats["contraction_ratio"] = round(len(all_contractions) / total_tokens, 6)

        # Average word length
        word_lengths = [len(t.text) for t in corpus_doc if t.is_alpha and not t.is_space]
        raw_stats["avg_word_length"] = round(
            float(np.mean(word_lengths)) if word_lengths else 0.0, 2
        )

        return raw_stats

    # =====================================================================
    # EMOJI ANALYSIS
    # =====================================================================

    def _compute_emoji_profile(self, tweets: list[CleanTweet]) -> EmojiProfile:
        """Compute emoji usage statistics from pre-extracted emoji positions."""
        if not tweets:
            return EmojiProfile()

        total_tweets = len(tweets)
        total_emoji_count = 0
        tweets_with_emoji = 0
        emoji_counter: Counter = Counter()
        position_buckets = {"leading": 0, "inline": 0, "trailing": 0}
        all_emojis_flat: list[str] = []

        # For pair co-occurrence
        emoji_pairs: Counter = Counter()

        for tweet in tweets:
            emojis_in_tweet = tweet.emoji_positions
            if emojis_in_tweet:
                tweets_with_emoji += 1

            tweet_emojis = []
            for ep in emojis_in_tweet:
                total_emoji_count += 1
                emoji_counter[ep.emoji] += 1
                all_emojis_flat.append(ep.emoji)
                tweet_emojis.append(ep.emoji)

                # Positional classification
                ratio = ep.position_ratio
                if ratio < 0.2:
                    position_buckets["leading"] += 1
                elif ratio > 0.8:
                    position_buckets["trailing"] += 1
                else:
                    position_buckets["inline"] += 1

            # Pair co-occurrence
            if len(tweet_emojis) >= 2:
                unique_sorted = sorted(set(tweet_emojis))
                for pair in combinations(unique_sorted, 2):
                    emoji_pairs[pair] += 1

        # --- Build profile ---
        usage_rate = total_emoji_count / total_tweets if total_tweets else 0.0
        emoji_tweet_ratio = tweets_with_emoji / total_tweets if total_tweets else 0.0

        top_emojis = emoji_counter.most_common(30)

        # Positional tendency
        total_pos = sum(position_buckets.values())
        if total_pos > 0:
            positional_tendency = {
                k: round(v / total_pos, 4) for k, v in position_buckets.items()
            }
        else:
            positional_tendency = {"leading": 0.0, "inline": 0.0, "trailing": 0.0}

        # Diversity
        unique_emojis = len(emoji_counter)
        emoji_diversity = (
            unique_emojis / total_emoji_count if total_emoji_count > 0 else 0.0
        )

        # Emoji sentiment (using emosent-py if available)
        emoji_sentiment = 0.0
        try:
            from emosent import get_emoji_sentiment

            sentiments = []
            for e in all_emojis_flat:
                try:
                    score = get_emoji_sentiment(e)
                    if score is not None:
                        sentiments.append(float(score.get("sentiment_score", 0.0)))
                except Exception:
                    pass
            if sentiments:
                emoji_sentiment = float(np.mean(sentiments))
        except ImportError:
            # Try alternative: compute a rough sentiment from known emoji mappings
            logger.debug("emosent not available; emoji sentiment set to 0.")

        return EmojiProfile(
            usage_rate=round(usage_rate, 4),
            emoji_tweet_ratio=round(emoji_tweet_ratio, 4),
            top_emojis=top_emojis,
            positional_tendency=positional_tendency,
            emoji_diversity=round(emoji_diversity, 4),
            emoji_sentiment=round(emoji_sentiment, 4),
        )

    # =====================================================================
    # HASHTAG ANALYSIS
    # =====================================================================

    def _compute_hashtag_profile(self, tweets: list[CleanTweet]) -> HashtagProfile:
        """Compute hashtag usage patterns."""
        if not tweets:
            return HashtagProfile()

        total_tweets = len(tweets)
        total_hashtags = 0
        hashtag_counter: Counter = Counter()
        trailing_count = 0
        inline_count = 0
        casing_counter: Counter = Counter()  # CamelCase, lowercase, UPPERCASE, mixed

        for tweet in tweets:
            tags = tweet.hashtags
            if not tags:
                continue

            total_hashtags += len(tags)

            for tag in tags:
                hashtag_counter[tag.lower()] += 1

                # Casing classification
                if tag.isupper():
                    casing_counter["UPPERCASE"] += 1
                elif tag.islower():
                    casing_counter["lowercase"] += 1
                elif tag[0].isupper() and any(c.isupper() for c in tag[1:]):
                    casing_counter["CamelCase"] += 1
                elif tag[0].isupper():
                    casing_counter["CamelCase"] += 1
                else:
                    casing_counter["mixed"] += 1

            # Placement: check if hashtags appear at the end of the tweet text
            text = tweet.text
            # Simple heuristic: if the last part of the tweet is all hashtags
            words = text.split()
            if words:
                # Count trailing hashtag-like tokens
                trailing_tags = 0
                for w in reversed(words):
                    if w.startswith("#"):
                        trailing_tags += 1
                    else:
                        break
                if trailing_tags == len(tags):
                    trailing_count += 1
                elif trailing_tags == 0:
                    inline_count += 1
                else:
                    # mixed
                    pass

        tweets_with_tags = sum(1 for t in tweets if t.hashtags)

        # Placement style
        if tweets_with_tags > 0:
            trailing_ratio = trailing_count / tweets_with_tags
            inline_ratio = inline_count / tweets_with_tags
            if trailing_ratio > 0.8:
                placement_style = "trailing"
            elif inline_ratio > 0.8:
                placement_style = "inline"
            else:
                placement_style = "mixed"
        else:
            placement_style = "mixed"

        # Casing style
        if casing_counter:
            dominant_casing = casing_counter.most_common(1)[0]
            total_cased = sum(casing_counter.values())
            if dominant_casing[1] / total_cased > 0.6:
                casing_style = dominant_casing[0]
            else:
                casing_style = "mixed"
        else:
            casing_style = "mixed"

        return HashtagProfile(
            usage_rate=round(total_hashtags / total_tweets if total_tweets else 0.0, 4),
            top_hashtags=hashtag_counter.most_common(20),
            placement_style=placement_style,
            casing_style=casing_style,
        )
