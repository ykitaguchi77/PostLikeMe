"""Text preprocessing and normalisation utilities.

These functions are used throughout the analysis pipeline to clean,
normalise, and extract features from raw tweet text.
"""

from __future__ import annotations

import re
import unicodedata

import emoji as emoji_lib

from postlikeme.utils.constants import CONTRACTION_MAP

# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once at import time for performance)
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
_TCO_PATTERN = re.compile(r"https?://t\.co/\w+", re.IGNORECASE)
_MENTION_PATTERN = re.compile(r"@(\w{1,15})")
_MULTI_SPACE = re.compile(r"\s+")
_CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP) + r")\b",
    re.IGNORECASE,
)

# Simple vowel set for syllable counting heuristic
_VOWELS = set("aeiouy")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Apply standard text normalisation for analysis.

    Steps performed:
    1. Unicode NFC normalisation (compose combining characters).
    2. Remove ``t.co`` shortened URLs.
    3. Collapse runs of whitespace to a single space.
    4. Strip leading / trailing whitespace.

    The function intentionally does **not** lowercase or remove emojis so
    that downstream analysers can make their own decisions.

    Parameters
    ----------
    text:
        Raw tweet text.

    Returns
    -------
    str
        Normalised text.
    """
    text = unicodedata.normalize("NFC", text)
    text = _TCO_PATTERN.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def extract_emojis(text: str) -> list[dict]:
    """Find every emoji in *text* with its position and normalised location.

    Uses the ``emoji`` library which covers the full Unicode emoji set
    including skin-tone modifiers, ZWJ sequences, and flag sequences.

    Parameters
    ----------
    text:
        The (possibly cleaned) tweet text.

    Returns
    -------
    list[dict]
        Each dict has keys:
        - ``emoji`` (str): the emoji character(s)
        - ``match_start`` (int): start index in the string
        - ``match_end`` (int): end index in the string
        - ``position_ratio`` (float): normalised position 0.0..1.0
    """
    text_len = max(len(text), 1)  # Avoid division by zero
    results: list[dict] = []
    for entry in emoji_lib.emoji_list(text):
        start = entry["match_start"]
        results.append(
            {
                "emoji": entry["emoji"],
                "match_start": start,
                "match_end": entry["match_end"],
                "position_ratio": start / text_len,
            }
        )
    return results


def count_syllables(word: str) -> int:
    """Estimate the number of syllables in an English word.

    Uses a simple vowel-group heuristic that is sufficient for aggregate
    readability metrics.  For individual-word precision, a pronunciation
    dictionary (e.g. CMUdict) would be needed.

    Rules:
    1. Count groups of consecutive vowels as one syllable each.
    2. If the word ends in a silent ``e``, subtract one (but never below 1).
    3. Ensure the result is at least 1.

    Parameters
    ----------
    word:
        A single English word (ASCII letters expected).

    Returns
    -------
    int
        Estimated syllable count (>= 1).
    """
    word = word.lower().strip()
    if not word:
        return 0

    count = 0
    prev_is_vowel = False

    for char in word:
        is_vowel = char in _VOWELS
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel

    # Silent-e adjustment
    if word.endswith("e") and count > 1:
        count -= 1

    # Special endings that add a syllable
    if word.endswith("le") and len(word) > 2 and word[-3] not in _VOWELS:
        count += 1

    return max(count, 1)


def is_all_caps(token: str) -> bool:
    """Return ``True`` if *token* is in ALL CAPS (at least 2 alpha characters).

    Single characters (e.g. "I") and non-alphabetic tokens are excluded
    to avoid false positives.

    Parameters
    ----------
    token:
        A single token / word.

    Returns
    -------
    bool
    """
    alpha_chars = [c for c in token if c.isalpha()]
    return len(alpha_chars) >= 2 and all(c.isupper() for c in alpha_chars)


def detect_contractions(text: str) -> list[str]:
    """Find all English contractions present in *text*.

    Contractions are detected using the ``CONTRACTION_MAP`` dictionary
    which covers standard English contractions.

    Parameters
    ----------
    text:
        Input text (case-insensitive matching is used).

    Returns
    -------
    list[str]
        Contractions found, in their original casing as they appear in the text.
    """
    return [match.group(0) for match in _CONTRACTION_PATTERN.finditer(text)]


def strip_urls(text: str) -> str:
    """Remove all URLs (http/https) from *text*.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    str
        Text with URLs removed and excess whitespace collapsed.
    """
    result = _URL_PATTERN.sub("", text)
    return _MULTI_SPACE.sub(" ", result).strip()


def strip_mentions(text: str) -> str:
    """Remove all @mentions from *text*.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    str
        Text with @mentions removed and excess whitespace collapsed.
    """
    result = _MENTION_PATTERN.sub("", text)
    return _MULTI_SPACE.sub(" ", result).strip()


def replace_mentions_with_placeholder(text: str) -> tuple[str, list[str]]:
    """Replace @mentions with ``@USER`` and return the originals.

    This is used during preprocessing so that mention usernames do not
    pollute vocabulary statistics, while the original mentions are
    preserved for other analysis steps.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    tuple[str, list[str]]
        A 2-tuple of (normalised text, list of original usernames without @).
    """
    originals: list[str] = []

    def _replace(match: re.Match) -> str:
        originals.append(match.group(1))
        return "@USER"

    normalised = _MENTION_PATTERN.sub(_replace, text)
    return normalised, originals


def strip_emojis(text: str) -> str:
    """Remove all emoji characters from *text*.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    str
        Text with emojis removed and excess whitespace collapsed.
    """
    result = emoji_lib.replace_emoji(text, replace="")
    return _MULTI_SPACE.sub(" ", result).strip()
