"""
app.py  –  Music Sentiment Analysis Dashboard
----------------------------------------------
Run with:  streamlit run app.py

Expects the following layout (all relative to this file):
    outputs/
        analyzed_music_posts.csv   ← produced by your crawler / analysis script
        sentiment_distribution.png
        top_common_words.png
        sentiment_over_time.png    ← optional, shown only if present
        topic_sentiment.png        ← optional
        wordcloud.png              ← optional

The CSV must have at minimum these columns:
    clean_text, sentiment, datetime

textblob_sentiment is computed on-the-fly here if the column is missing.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from nlp_pipeline import (
    textblob_sentiment,
    tfidf_top_words,
    tfidf_by_sentiment,
)

# ── Config ───────────────────────────────────────────────────────────────────

OUTPUTS = Path("outputs")

st.set_page_config(
    page_title="Music Sentiment Analysis Dashboard",
    layout="wide",
)

st.title("Music Discussion Sentiment Analysis Pipeline")
st.caption("Data sourced from 4chan /mu/ · NLP: VADER + TextBlob · Vectorisation: TF-IDF")

# ── Load data ────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "analyzed_music_posts.csv")

    # Add textblob_sentiment if the CSV doesn't have it yet
    if "textblob_sentiment" not in df.columns:
        with st.spinner("Computing TextBlob sentiment (first run only)…"):
            df["textblob_sentiment"] = df["clean_text"].fillna("").apply(
                textblob_sentiment
            )

    # Parse datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["date"] = df["datetime"].dt.date

    return df


df = load_data()

# ── Overview metrics ─────────────────────────────────────────────────────────

st.subheader("Dataset overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total posts", f"{len(df):,}")
with col2:
    agree = (df["sentiment"] == df["textblob_sentiment"]).mean()
    st.metric("VADER / TextBlob agreement", f"{agree:.1%}")
with col3:
    top_sent = df["sentiment"].value_counts().idxmax()
    st.metric("Dominant sentiment", top_sent.capitalize())
with col4:
    if "date" in df.columns:
        span = df["datetime"].dropna()
        days = (span.max() - span.min()).days
        st.metric("Date span (days)", days)

# ── Sentiment distribution ───────────────────────────────────────────────────

st.subheader("Sentiment distribution")
sentiment_counts = df["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Posts")
st.bar_chart(sentiment_counts.set_index("Sentiment"))

# ── VADER vs TextBlob ────────────────────────────────────────────────────────

st.subheader("VADER vs TextBlob comparison")
comparison = pd.crosstab(
    df["sentiment"],
    df["textblob_sentiment"],
    rownames=["VADER"],
    colnames=["TextBlob"],
)
st.dataframe(comparison, use_container_width=True)

# ── TF-IDF word analysis ─────────────────────────────────────────────────────

st.subheader("Top words by TF-IDF score")
st.caption(
    "TF-IDF surfaces distinctive words per sentiment class — not just the "
    "most frequent overall (which are mostly stop words)."
)

tfidf_tab1, tfidf_tab2 = st.tabs(["Overall", "By sentiment"])

with tfidf_tab1:
    top_overall = tfidf_top_words(df["clean_text"], n=20)
    if not top_overall.empty:
        st.bar_chart(top_overall.set_index("word")["tfidf_score"])

with tfidf_tab2:
    per_class = tfidf_by_sentiment(df, n=10)
    cols = st.columns(len(per_class))
    for col, (label, frame) in zip(cols, per_class.items()):
        with col:
            st.markdown(f"**{label.capitalize()}**")
            if not frame.empty:
                st.bar_chart(frame.set_index("word")["tfidf_score"])

# ── Sentiment over time ───────────────────────────────────────────────────────

if "date" in df.columns and df["date"].notna().any():
    st.subheader("Sentiment over time")
    time_df = (
        df.groupby(["date", "sentiment"])
        .size()
        .reset_index(name="count")
        .pivot(index="date", columns="sentiment", values="count")
        .fillna(0)
    )
    st.line_chart(time_df)

# ── Static visualisations (optional, shown only if files exist) ──────────────

image_specs = [
    ("sentiment_distribution.png", "Sentiment distribution"),
    ("top_common_words.png",        "Top common words (raw counts)"),
    ("sentiment_over_time.png",     "Sentiment trends over time"),
    ("topic_sentiment.png",         "Topic sentiment"),
    ("wordcloud.png",               "Word cloud"),
]

available = [(fn, cap) for fn, cap in image_specs if (OUTPUTS / fn).exists()]

if available:
    st.subheader("Saved visualisations")
    for i in range(0, len(available), 2):
        pair = available[i : i + 2]
        cols = st.columns(len(pair))
        for col, (fn, cap) in zip(cols, pair):
            col.image(str(OUTPUTS / fn), caption=cap)

# ── Sample posts ─────────────────────────────────────────────────────────────

st.subheader("Sample analysed posts")

sentiment_filter = st.selectbox(
    "Filter by sentiment",
    ["All"] + sorted(df["sentiment"].dropna().unique().tolist()),
)

display_df = df if sentiment_filter == "All" else df[df["sentiment"] == sentiment_filter]

display_cols = ["clean_text", "sentiment", "textblob_sentiment"]
if "date" in display_df.columns:
    display_cols = ["date"] + display_cols
available_cols = [c for c in display_cols if c in display_df.columns]

st.dataframe(
    display_df[available_cols].dropna(subset=["clean_text"]).head(50),
    use_container_width=True,
)
