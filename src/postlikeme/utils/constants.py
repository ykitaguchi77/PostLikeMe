"""Project-wide constants used across the analysis and generation pipeline.

These word lists are derived from established psycholinguistic research
(LIWC categories, Pennebaker & King 1999, Yarkoni 2010) and adapted for
social-media analysis.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Function words -- the 50 most common English function words.
# These form the backbone of stylometric "fingerprinting" because authors
# use them unconsciously and consistently regardless of topic.
# ---------------------------------------------------------------------------

FUNCTION_WORDS: list[str] = [
    "the", "of", "and", "to", "a", "in", "that", "it", "is", "was",
    "for", "with", "as", "but", "be", "on", "not", "he", "she", "this",
    "from", "by", "at", "or", "an", "which", "have", "had", "were", "been",
    "do", "if", "so", "than", "its", "my", "into", "about", "what", "their",
    "who", "could", "would", "should", "just", "also", "very", "even", "too",
    "all",
]

# ---------------------------------------------------------------------------
# Slang words -- informal vocabulary common on social media.
# Used to compute formality score; higher slang ratio = more casual.
# ---------------------------------------------------------------------------

SLANG_WORDS: set[str] = {
    "lol", "lmao", "lmfao", "rofl", "smh", "tbh", "imo", "imho", "irl",
    "fomo", "yolo", "fwiw", "tl;dr", "tldr", "ngl", "idk", "idgaf", "af",
    "bruh", "bro", "dude", "fam", "lit", "slay", "vibe", "vibes", "lowkey",
    "highkey", "goat", "goated", "sus", "cap", "no cap", "bet", "fr",
    "deadass", "salty", "shade", "tea", "stan", "simp", "flex", "ghosting",
    "clout", "bussin", "fire", "mid", "based", "cringe", "ratio", "w",
    "ong", "nah", "gonna", "gotta", "wanna", "kinda", "sorta", "lemme",
    "dunno", "ain't", "ya", "yall", "y'all", "cuz", "tho", "thx",
    "pls", "plz", "omg", "omfg", "wtf", "wth", "stfu", "gtfo", "ikr",
    "rn", "dm", "hmu", "ftw", "smth", "sth",
}

# ---------------------------------------------------------------------------
# Swear words -- basic set for agreeableness scoring.
# Research shows swear-word frequency has the strongest single correlation
# with low agreeableness (r = -0.39, Yarkoni 2010).
# ---------------------------------------------------------------------------

SWEAR_WORDS: set[str] = {
    "damn", "dammit", "hell", "crap", "crapola", "crapshoot",
    "shit", "shitty", "bullshit", "horseshit", "batshit",
    "fuck", "fucking", "fucked", "fucker", "motherfucker",
    "ass", "asshole", "badass", "dumbass", "jackass",
    "bitch", "bitchy", "bastard",
    "dick", "dickhead", "piss", "pissed",
    "goddam", "goddamn", "goddammit",
    "wtf", "stfu", "gtfo", "lmfao",
}

# ---------------------------------------------------------------------------
# Achievement words -- markers of conscientiousness.
# ---------------------------------------------------------------------------

ACHIEVEMENT_WORDS: set[str] = {
    "achieve", "achieved", "achievement", "accomplish", "accomplished",
    "complete", "completed", "finish", "finished", "goal", "goals",
    "success", "successful", "succeed", "win", "won", "earn", "earned",
    "improve", "improved", "improvement", "progress", "milestone",
    "deliver", "delivered", "build", "built", "create", "created",
    "launch", "launched", "ship", "shipped", "result", "results",
    "productive", "productivity", "efficient", "efficiency",
    "excellence", "outstanding", "perfect", "mastered", "master",
    "determined", "discipline", "focused", "driven", "ambition",
    "ambitious", "diligent", "thorough",
}

# ---------------------------------------------------------------------------
# Anxiety words -- markers of neuroticism.
# ---------------------------------------------------------------------------

ANXIETY_WORDS: set[str] = {
    "worried", "worry", "worrying", "anxious", "anxiety", "nervous",
    "afraid", "fear", "feared", "fearful", "scared", "scary",
    "panic", "panicked", "panicking", "stressed", "stress", "stressful",
    "tense", "tension", "uneasy", "dread", "dreading", "terrified",
    "terrifying", "overwhelmed", "overwhelming", "restless",
    "insecure", "insecurity", "uncertain", "uncertainty",
    "paranoid", "paranoia", "alarmed", "alarming", "apprehensive",
    "distressed", "distressing", "frantic", "desperate",
    "helpless", "hopeless", "vulnerable",
}

# ---------------------------------------------------------------------------
# Social words -- markers of extraversion.
# ---------------------------------------------------------------------------

SOCIAL_WORDS: set[str] = {
    "friend", "friends", "friendship", "buddy", "pal", "mate",
    "party", "parties", "hang", "hangout", "together", "meet",
    "meeting", "met", "team", "group", "club", "community",
    "people", "everyone", "everybody", "folk", "folks",
    "social", "socialize", "chat", "talk", "talked", "talking",
    "conversation", "discuss", "discussed", "share", "shared",
    "celebrate", "celebrated", "celebration", "gather", "gathering",
    "invite", "invited", "join", "joined", "connect", "connected",
    "connection", "network", "networking", "collab", "collaborate",
    "family", "relationship", "relationships", "love", "loved",
}

# ---------------------------------------------------------------------------
# Filler words -- negative markers of conscientiousness.
# ---------------------------------------------------------------------------

FILLER_WORDS: set[str] = {
    "like", "literally", "basically", "actually", "honestly",
    "anyway", "anyways", "whatever", "whatnot", "somehow",
    "stuff", "things", "thing", "kind of", "sort of",
    "you know", "i mean", "i guess", "i think", "i feel like",
    "maybe", "perhaps", "probably", "supposedly", "apparently",
    "um", "uh", "er", "hmm", "hm", "mhm", "huh",
    "well", "so", "right", "ok", "okay",
}

# ---------------------------------------------------------------------------
# Certainty words -- positive markers of conscientiousness.
# ---------------------------------------------------------------------------

CERTAINTY_WORDS: set[str] = {
    "certainly", "definitely", "absolutely", "always", "never",
    "undoubtedly", "clearly", "obviously", "surely", "without doubt",
    "guaranteed", "inevitable", "inevitably", "unquestionably",
    "precisely", "exactly", "must", "certain", "confident",
    "convinced", "sure", "positive", "conclusive", "conclusively",
    "indisputable", "undeniable", "undeniably", "assuredly",
    "decidedly", "emphatically", "firmly",
}

# ---------------------------------------------------------------------------
# Contraction map -- for detecting contraction usage preferences.
# Keys are the contracted forms; values are the expanded equivalents.
# ---------------------------------------------------------------------------

CONTRACTION_MAP: dict[str, str] = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "mustn't": "must not",
    "must've": "must have",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# ---------------------------------------------------------------------------
# Numeric constants
# ---------------------------------------------------------------------------

DEFAULT_HALF_LIFE_DAYS: int = 90
"""Half-life in days for the recency exponential-decay weighting of topics.
A tweet 90 days old gets weight 0.5, 180 days old gets 0.25, etc."""

MIN_TWEETS_FOR_ANALYSIS: int = 100
"""Minimum number of tweets required to run the analysis pipeline.
Below this threshold a warning is emitted and only basic structural
analysis is performed."""

RECOMMENDED_TWEETS: int = 500
"""Recommended number of tweets for a high-quality voice profile."""

MAX_TWEET_LENGTH: int = 280
"""Maximum character length for a single tweet on X/Twitter."""
