"""
nlp_pipeline.py
---------------
Preprocessing, sentiment scoring, and TF-IDF analysis for /mu/ posts.

Replaces ad-hoc cleaning scattered in older scripts with a single module
that every other file imports.

Concepts applied
----------------
* Text processing  – tokenisation, lemmatisation, stopword removal
* Vectorisation    – TF-IDF (replaces raw word counts in the dashboard)
* MLOps lifecycle  – functions are stateless and importable; heavy
                     one-off work (NLTK downloads) is guarded so the
                     module is safe to import in a Faktory worker.
"""

import re
import string
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── one-time NLTK downloads (no-ops if already cached) ──────────────────────
for pkg in ("punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"):
    nltk.download(pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()
_vader = SentimentIntensityAnalyzer()

# Extended stopword set: NLTK base + board-specific noise
_EXTRA_STOPS = {
    "music", "song", "songs", "album", "albums", "artist", "artists",
    "band", "bands", "listen", "listening", "like", "good", "really",
    "just", "know", "think", "people", "thats", "dont", "im", "ive",
    "get", "got", "even", "shit", "amp", "gt", "lt", "br", "www",
    "http", "https", "com", "org", "mu", "thread", "post", "board",
}
_STOPS = stopwords.words("english")
_STOPS = set(_STOPS) | _EXTRA_STOPS


# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_html(text: str) -> str:
    """Strip HTML tags and decode common entities."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#039;", "'")
            .replace("&nbsp;", " ")
    )
    return text.strip()


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline:
      1. Strip HTML
      2. Lowercase
      3. Remove URLs
      4. Remove punctuation / digits
      5. Tokenise
      6. Remove stopwords
      7. Lemmatise
    """
    text = clean_html(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _STOPS and len(t) > 2
    ]
    return " ".join(tokens)


# ── Sentiment scoring ────────────────────────────────────────────────────────

def vader_sentiment(text: str) -> str:
    """Return 'positive', 'negative', or 'neutral' via VADER compound score."""
    score = _vader.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def vader_score(text: str) -> float:
    """Return raw VADER compound score (−1 … +1)."""
    return _vader.polarity_scores(str(text))["compound"]


def textblob_sentiment(text: str) -> str:
    """Return 'positive', 'negative', or 'neutral' via TextBlob polarity."""
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.05:
        return "positive"
    if polarity < -0.05:
        return "negative"
    return "neutral"


# ── TF-IDF analysis ──────────────────────────────────────────────────────────

def tfidf_top_words(
    corpus: pd.Series,
    n: int = 20,
    ngram_range: tuple = (1, 1),
) -> pd.DataFrame:
    """
    Fit TF-IDF on a corpus and return the top-n terms by mean TF-IDF score.

    TF-IDF surfaces *distinctive* words — terms that appear often in a
    document but rarely across the whole corpus — rather than just the most
    frequent words overall (which are mostly stopwords even after filtering).

    Parameters
    ----------
    corpus      : pd.Series of cleaned strings
    n           : how many top terms to return
    ngram_range : (1,1) for unigrams, (1,2) to include bigrams

    Returns
    -------
    pd.DataFrame with columns ['word', 'tfidf_score']
    """
    corpus = corpus.dropna().astype(str)
    corpus = corpus[corpus.str.strip() != ""]
    if len(corpus) == 0:
        return pd.DataFrame(columns=["word", "tfidf_score"])

    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=ngram_range,
        min_df=2,          # ignore terms that appear in < 2 docs
        sublinear_tf=True, # apply log(1 + tf) to dampen high frequencies
    )
    matrix = vec.fit_transform(corpus)
    scores = matrix.mean(axis=0).A1
    terms = vec.get_feature_names_out()
    top_idx = scores.argsort()[::-1][:n]
    return pd.DataFrame({
        "word": terms[top_idx],
        "tfidf_score": scores[top_idx],
    })


def tfidf_by_sentiment(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    sentiment_col: str = "sentiment",
    n: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Return top-n TF-IDF terms *per sentiment class*.

    Useful for understanding what language drives each class rather than
    what's globally common.
    """
    results = {}
    for label in df[sentiment_col].dropna().unique():
        subset = df.loc[df[sentiment_col] == label, text_col]
        results[label] = tfidf_top_words(subset, n=n)
    return results


# ── Word frequency (kept for backwards compat) ──────────────────────────────

def top_words_raw(corpus: pd.Series, n: int = 20) -> pd.DataFrame:
    """Raw token frequency — use tfidf_top_words for meaningful results."""
    all_tokens: list[str] = []
    for doc in corpus.dropna():
        all_tokens.extend(str(doc).split())
    counts = Counter(all_tokens).most_common(n)
    return pd.DataFrame(counts, columns=["word", "count"])
