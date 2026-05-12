"""
nlp_pipeline.py
---------------
Preprocessing, sentiment scoring, and TF-IDF analysis for EchoMood:
A Scalable NLP Pipeline for Music Discussion Sentiment Analysis.

Replaces ad-hoc cleaning scattered across older scripts with a unified,
reusable NLP module imported throughout the project pipeline.

Concepts applied
----------------
* Text processing  – tokenization, lemmatization, stopword removal
* Vectorization    – TF-IDF feature extraction
* Sentiment NLP    – VADER + TextBlob polarity scoring
* MLOps lifecycle  – stateless, importable functions safe for async workers
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

# ── One-time NLTK downloads (safe if already cached) ────────────────────────

for pkg in ("punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"):
    nltk.download(pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()
_vader = SentimentIntensityAnalyzer()

# Extended stopword set: NLTK defaults + board-specific noise words
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
    """Remove HTML tags and decode common entities."""
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
    Full NLP preprocessing pipeline:

      1. Strip HTML
      2. Lowercase conversion
      3. Remove URLs
      4. Remove punctuation and digits
      5. Tokenize
      6. Remove stopwords
      7. Lemmatize
    """

    text = clean_html(text)
    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = word_tokenize(text)

    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _STOPS and len(t) > 2
    ]

    return " ".join(tokens)

# ── Sentiment scoring ────────────────────────────────────────────────────────

def vader_sentiment(text: str) -> str:
    """
    Return sentiment label using VADER compound score.

    Labels:
        positive
        negative
        neutral
    """

    score = _vader.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"

    if score <= -0.05:
        return "negative"

    return "neutral"


def vader_score(text: str) -> float:
    """Return raw VADER compound score between -1 and +1."""
    return _vader.polarity_scores(str(text))["compound"]


def textblob_sentiment(text: str) -> str:
    """
    Return sentiment label using TextBlob polarity analysis.
    """

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
    Fit TF-IDF on a corpus and return top-ranked terms.

    TF-IDF highlights distinctive vocabulary rather than simply
    the most frequent words in the dataset.
    """

    corpus = corpus.dropna().astype(str)
    corpus = corpus[corpus.str.strip() != ""]

    if len(corpus) == 0:
        return pd.DataFrame(columns=["word", "tfidf_score"])

    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
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
    Compute top TF-IDF terms separately for each sentiment class.
    """

    results = {}

    for label in df[sentiment_col].dropna().unique():
        subset = df.loc[df[sentiment_col] == label, text_col]
        results[label] = tfidf_top_words(subset, n=n)

    return results

# ── Raw word frequency (legacy support) ──────────────────────────────────────

def top_words_raw(corpus: pd.Series, n: int = 20) -> pd.DataFrame:
    """
    Raw token frequency counts.

    TF-IDF analysis is generally preferred for meaningful insights.
    """

    all_tokens: list[str] = []

    for doc in corpus.dropna():
        all_tokens.extend(str(doc).split())

    counts = Counter(all_tokens).most_common(n)

    return pd.DataFrame(counts, columns=["word", "count"])