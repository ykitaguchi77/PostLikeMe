"""CLI command for analysing collected tweets and building a voice profile.

Usage::

    postlikeme analyze <username> [--rebuild] [--skip-llm]
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def analyze(
    username: str = typer.Argument(
        ...,
        help="X/Twitter username to analyse (must already be collected).",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        "-r",
        help="Force rebuild even if a profile already exists.",
    ),
    skip_llm: bool = typer.Option(
        False,
        "--skip-llm",
        help="Skip LLM-powered analysis steps (faster but less accurate personality).",
    ),
) -> None:
    """Analyse collected tweets and build a voice profile."""
    username = username.lstrip("@").lower()

    from postlikeme.models.config import AppConfig
    from postlikeme.storage.config_store import ConfigStore
    from postlikeme.storage.profile_store import ProfileStore

    config_store = ConfigStore(AppConfig().data_dir)
    config = config_store.load()
    config.ensure_dirs()

    profile_store = ProfileStore(config.data_dir)

    # Check for existing profile
    if not rebuild:
        existing = profile_store.load_profile(username)
        if existing is not None:
            console.print(
                f"[yellow]Profile for @{username} already exists "
                f"(built {existing.profile_built_at.strftime('%Y-%m-%d %H:%M UTC')}). "
                f"Use --rebuild to regenerate.[/yellow]"
            )
            raise typer.Exit(0)

    # Load cached tweets
    cache_file = config.cache_dir / f"{username}_tweets.json"
    if not cache_file.exists():
        console.print(
            f"[red]No cached tweets found for @{username}. "
            f"Run 'postlikeme collect {username}' first.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"\n[bold]Analysing tweets for @{username}[/bold]\n")

    tweets_data = json.loads(cache_file.read_text(encoding="utf-8"))

    if not tweets_data:
        console.print("[red]Cache file is empty. Re-collect tweets first.[/red]")
        raise typer.Exit(1)

    from postlikeme.utils.constants import MIN_TWEETS_FOR_ANALYSIS, RECOMMENDED_TWEETS

    if len(tweets_data) < MIN_TWEETS_FOR_ANALYSIS:
        console.print(
            f"[yellow]Warning: Only {len(tweets_data)} tweets found. "
            f"Minimum recommended: {MIN_TWEETS_FOR_ANALYSIS}. "
            f"Optimal: {RECOMMENDED_TWEETS}+. Profile quality may be limited.[/yellow]"
        )

    try:
        profile = asyncio.run(
            _run_analysis(username, tweets_data, config, skip_llm)
        )
    except Exception as exc:
        console.print(f"[red]Analysis failed: {exc}[/red]")
        raise typer.Exit(1)

    # Save profile
    saved_path = profile_store.save_profile(profile)
    console.print(f"\n[green]Profile saved to: {saved_path}[/green]")

    # Display summary
    _display_profile_summary(profile)


async def _run_analysis(
    username: str,
    tweets_data: list[dict],
    config: object,
    skip_llm: bool,
) -> object:
    """Run the analysis pipeline and return a VoiceProfile.

    This function performs structural, emoji, hashtag, topic, and personality
    analysis on the collected tweets.
    """
    from datetime import datetime

    from postlikeme.models.voice_profile import (
        BigFiveScores,
        CapitalizationPatterns,
        EmojiProfile,
        ExampleTweets,
        HashtagProfile,
        OpinionEntry,
        PersonalityProfile,
        PunctuationPatterns,
        ReadabilityScores,
        ReplyStyle,
        StructuralStyle,
        TopicEntry,
        TopicProfile,
        TweetLengthDistribution,
        VoiceProfile,
    )
    from postlikeme.utils.text import extract_emojis, normalize_text

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # ---- Step 1: Preprocess tweets ----
        task = progress.add_task("Preprocessing tweets...", total=None)

        # Separate originals and replies
        originals: list[dict] = []
        replies: list[dict] = []
        all_texts: list[str] = []

        for t in tweets_data:
            text = t.get("text", "")
            if not text or t.get("is_retweet", False):
                continue
            cleaned_text = normalize_text(text)
            if not cleaned_text:
                continue

            all_texts.append(cleaned_text)
            if t.get("is_reply", False):
                replies.append(t)
            else:
                originals.append(t)

        progress.update(task, completed=True)

        # ---- Step 2: Structural analysis ----
        task = progress.add_task("Analysing structural style...", total=None)

        lengths = [len(normalize_text(t.get("text", ""))) for t in originals if t.get("text")]
        if not lengths:
            lengths = [len(t) for t in all_texts] if all_texts else [0]

        import statistics

        avg_len = statistics.mean(lengths) if lengths else 0
        median_len = statistics.median(lengths) if lengths else 0
        std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0

        # Sentence-level analysis
        sentence_lengths: list[int] = []
        question_count = 0
        exclamation_count = 0
        fragment_count = 0
        total_sentences = 0

        for text in all_texts:
            import re

            sentences = re.split(r"[.!?]+", text)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                total_sentences += 1
                words_in_sent = sent.split()
                sentence_lengths.append(len(words_in_sent))
                if len(words_in_sent) <= 3:
                    fragment_count += 1

            question_count += text.count("?")
            exclamation_count += text.count("!")

        avg_sentence_len = (
            statistics.mean(sentence_lengths) if sentence_lengths else 0
        )
        fragment_ratio = fragment_count / max(total_sentences, 1)
        question_ratio = question_count / max(total_sentences, 1)
        exclamation_ratio = exclamation_count / max(total_sentences, 1)

        # Capitalization analysis
        all_words = " ".join(all_texts).split()
        all_caps_count = sum(
            1 for w in all_words if len(w) >= 2 and w.isalpha() and w.isupper()
        )
        all_caps_ratio = all_caps_count / max(len(all_words), 1)

        lowercase_starts = sum(
            1 for t in all_texts if t and t[0].islower()
        )
        lowercase_start_ratio = lowercase_starts / max(len(all_texts), 1)

        # Length distribution
        sorted_lengths = sorted(lengths)
        p25 = sorted_lengths[len(sorted_lengths) // 4] if sorted_lengths else 0
        p75 = sorted_lengths[3 * len(sorted_lengths) // 4] if sorted_lengths else 0

        histogram: dict[str, int] = {}
        for l_val in lengths:
            bucket = f"{(l_val // 50) * 50}-{(l_val // 50) * 50 + 49}"
            histogram[bucket] = histogram.get(bucket, 0) + 1

        tweet_length_dist = TweetLengthDistribution(
            mean=avg_len,
            median=median_len,
            std=std_len,
            min=min(lengths) if lengths else 0,
            max=max(lengths) if lengths else 0,
            p25=float(p25),
            p75=float(p75),
            histogram=histogram,
        )

        structural = StructuralStyle(
            avg_tweet_length=avg_len,
            tweet_length_distribution=tweet_length_dist,
            avg_sentence_length=avg_sentence_len,
            fragment_ratio=fragment_ratio,
            question_ratio=question_ratio,
            exclamation_ratio=exclamation_ratio,
            capitalization_patterns=CapitalizationPatterns(
                all_caps_ratio=all_caps_ratio,
                lowercase_start_ratio=lowercase_start_ratio,
                initial_caps_ratio=1.0 - lowercase_start_ratio,
            ),
            punctuation_patterns=PunctuationPatterns(),
            readability_scores=ReadabilityScores(),
        )

        progress.update(task, completed=True)

        # ---- Step 3: Emoji analysis ----
        task = progress.add_task("Analysing emoji patterns...", total=None)

        emoji_counts: dict[str, int] = {}
        total_emojis = 0
        tweets_with_emojis = 0

        for text in all_texts:
            emojis = extract_emojis(text)
            if emojis:
                tweets_with_emojis += 1
            for e in emojis:
                total_emojis += 1
                emoji_char = e["emoji"]
                emoji_counts[emoji_char] = emoji_counts.get(emoji_char, 0) + 1

        total_tweets = max(len(all_texts), 1)
        emoji_usage_rate = total_emojis / total_tweets
        emoji_tweet_ratio = tweets_with_emojis / total_tweets

        top_emojis = sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        unique_emojis = len(emoji_counts)
        emoji_diversity = unique_emojis / max(total_emojis, 1)

        emoji_prof = EmojiProfile(
            usage_rate=emoji_usage_rate,
            emoji_tweet_ratio=emoji_tweet_ratio,
            top_emojis=top_emojis,
            emoji_diversity=emoji_diversity,
        )

        progress.update(task, completed=True)

        # ---- Step 4: Hashtag analysis ----
        task = progress.add_task("Analysing hashtag patterns...", total=None)

        hashtag_counts: dict[str, int] = {}
        total_hashtags = 0

        for t in tweets_data:
            for ht in t.get("hashtags", []):
                if ht:
                    total_hashtags += 1
                    hashtag_counts[ht.lower()] = hashtag_counts.get(ht.lower(), 0) + 1

        hashtag_usage_rate = total_hashtags / total_tweets
        top_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        hashtag_prof = HashtagProfile(
            usage_rate=hashtag_usage_rate,
            top_hashtags=top_hashtags,
        )

        progress.update(task, completed=True)

        # ---- Step 5: Topic analysis ----
        task = progress.add_task("Identifying topics...", total=None)

        # Simple keyword-based topic detection
        from collections import Counter

        word_freq: Counter[str] = Counter()
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "about", "like",
            "through", "after", "over", "between", "out", "against", "during",
            "without", "before", "under", "around", "among", "it", "this",
            "that", "these", "those", "i", "you", "he", "she", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his", "its",
            "our", "their", "what", "which", "who", "when", "where", "how",
            "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "no", "not", "only", "same", "so", "than", "too",
            "very", "just", "because", "but", "and", "or", "if", "while",
            "although", "even", "also", "still", "already", "yet", "now",
            "then", "here", "there", "up", "down", "get", "got", "go",
            "going", "been", "really", "much", "well", "back", "don't",
            "i'm", "it's", "that's", "don't", "can't", "won't", "didn't",
        }

        for text in all_texts:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            for w in words:
                if w not in stopwords:
                    word_freq[w] += 1

        # Group frequent words into rough topic clusters
        top_words = word_freq.most_common(50)
        topics_ranked: list[TopicEntry] = []
        if top_words:
            max_count = top_words[0][1]
            # Take top 10 words as pseudo-topics
            for word, cnt in top_words[:10]:
                importance = cnt / max_count
                topics_ranked.append(
                    TopicEntry(
                        name=word.capitalize(),
                        importance=round(importance, 3),
                        keywords=[word],
                    )
                )

        topic_prof = TopicProfile(topics_ranked=topics_ranked)

        progress.update(task, completed=True)

        # ---- Step 6: Personality analysis ----
        task = progress.add_task("Assessing personality traits...", total=None)

        from postlikeme.utils.constants import SLANG_WORDS

        # Formality: based on slang usage
        all_text_lower = " ".join(all_texts).lower()
        all_word_list = all_text_lower.split()
        slang_count = sum(1 for w in all_word_list if w in SLANG_WORDS)
        slang_ratio = slang_count / max(len(all_word_list), 1)
        formality_score = max(0.0, min(1.0, 1.0 - slang_ratio * 10))

        # Tone inference
        if formality_score > 0.7:
            tone = "professional"
        elif formality_score > 0.4:
            tone = "casual"
        else:
            tone = "casual"

        # Humor style (simple heuristic)
        humor_style = "none"
        humor_markers = {
            "lol": "wholesome",
            "lmao": "absurdist",
            "rofl": "absurdist",
            "smh": "sarcastic",
            "bruh": "dry",
            "/s": "sarcastic",
            "haha": "wholesome",
            "lol": "wholesome",
        }
        humor_votes: dict[str, int] = {}
        for marker, style in humor_markers.items():
            cnt = all_text_lower.count(marker)
            if cnt > 0:
                humor_votes[style] = humor_votes.get(style, 0) + cnt
        if humor_votes:
            humor_style = max(humor_votes, key=humor_votes.get)  # type: ignore[arg-type]

        personality = PersonalityProfile(
            formality_score=round(formality_score, 3),
            humor_style=humor_style,
            tone=tone,
            assertiveness=0.5,
        )

        progress.update(task, completed=True)

        # ---- Step 7: Reply style ----
        task = progress.add_task("Analysing reply patterns...", total=None)

        reply_lengths = [
            len(normalize_text(r.get("text", "")))
            for r in replies
            if r.get("text")
        ]
        avg_reply_len = statistics.mean(reply_lengths) if reply_lengths else 0

        reply_style = ReplyStyle(
            avg_reply_length=avg_reply_len,
            length_ratio=avg_reply_len / max(avg_len, 1),
        )

        progress.update(task, completed=True)

        # ---- Step 8: Select example tweets ----
        task = progress.add_task("Selecting example tweets...", total=None)

        # Select diverse examples from originals
        original_texts = [
            normalize_text(t.get("text", ""))
            for t in originals
            if t.get("text") and not t.get("text", "").startswith("RT @")
        ]
        # Pick up to 25 examples, spread across the collection
        step = max(1, len(original_texts) // 25)
        example_originals = original_texts[::step][:25]

        # Select reply examples
        reply_pairs: list[tuple[str, str]] = []
        for r in replies[:20]:
            reply_text = normalize_text(r.get("text", ""))
            context = r.get("reply_to_user", "someone")
            if reply_text:
                reply_pairs.append((f"@{context}'s tweet", reply_text))

        example_tweets = ExampleTweets(
            originals=example_originals,
            replies=reply_pairs[:15],
        )

        progress.update(task, completed=True)

        # ---- Step 9: Build profile ----
        task = progress.add_task("Building voice profile...", total=None)

        # Date range
        dates = []
        for t in tweets_data:
            created = t.get("created_at", "")
            if created:
                try:
                    if isinstance(created, str):
                        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    else:
                        dt = created
                    dates.append(dt)
                except (ValueError, TypeError):
                    pass

        date_start = min(dates) if dates else None
        date_end = max(dates) if dates else None

        profile = VoiceProfile(
            username=username,
            tweet_count_analyzed=len(all_texts),
            date_range_start=date_start,
            date_range_end=date_end,
            structural_style=structural,
            emoji_profile=emoji_prof,
            hashtag_profile=hashtag_prof,
            topic_profile=topic_prof,
            personality_profile=personality,
            reply_style=reply_style,
            example_tweets=example_tweets,
        )

        progress.update(task, completed=True)

    return profile


def _display_profile_summary(profile: object) -> None:
    """Display a condensed summary of the built profile."""
    from postlikeme.models.voice_profile import VoiceProfile

    if not isinstance(profile, VoiceProfile):
        return

    table = Table(title=f"Voice Profile Summary: @{profile.username}", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Tweets analysed", str(profile.tweet_count_analyzed))

    if profile.date_range_start and profile.date_range_end:
        table.add_row(
            "Date range",
            f"{profile.date_range_start.strftime('%Y-%m-%d')} to "
            f"{profile.date_range_end.strftime('%Y-%m-%d')}",
        )

    table.add_row(
        "Avg tweet length",
        f"{profile.structural_style.avg_tweet_length:.0f} chars",
    )
    table.add_row("Tone", profile.personality_profile.tone)
    table.add_row(
        "Formality",
        f"{profile.personality_profile.formality_score:.2f}",
    )
    table.add_row("Humor style", profile.personality_profile.humor_style)
    table.add_row(
        "Emoji rate",
        f"{profile.emoji_profile.usage_rate:.2f} per tweet",
    )
    table.add_row(
        "Hashtag rate",
        f"{profile.hashtag_profile.usage_rate:.2f} per tweet",
    )

    if profile.topic_profile.topics_ranked:
        top_topics = ", ".join(
            t.name for t in profile.topic_profile.topics_ranked[:5]
        )
        table.add_row("Top topics", top_topics)

    table.add_row(
        "Example tweets",
        f"{len(profile.example_tweets.originals)} originals, "
        f"{len(profile.example_tweets.replies)} reply pairs",
    )

    console.print()
    console.print(table)
    console.print()
