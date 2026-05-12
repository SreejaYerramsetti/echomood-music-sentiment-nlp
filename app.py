"""
app.py  –  EchoMood Dashboard
----------------------------------------------
Run with:  streamlit run app.py

EchoMood:
A Scalable NLP Pipeline for Music Discussion Sentiment Analysis

Expects the following layout:

    outputs/
        analyzed_music_posts.csv
        sentiment_distribution.png
        top_common_words.png
        sentiment_over_time.png    ← optional
        topic_sentiment.png        ← optional
        wordcloud.png              ← optional

The CSV must have at minimum these columns:

    clean_text, sentiment, datetime

textblob_sentiment is computed on-the-fly if the column is missing.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from nlp_pipeline import (
    textblob_sentiment,
    tfidf_top_words,
    tfidf_by_sentiment,
)

# ── Config ───────────────────────────────────────────────────────────────────

PROJECT_TITLE = "EchoMood: A Scalable NLP Pipeline for Music Discussion Sentiment Analysis"
OUTPUTS = Path("outputs")

st.set_page_config(
    page_title="EchoMood",
    layout="wide",
)

st.title(PROJECT_TITLE)
st.caption(
    "Data sourced from 4chan /mu/ · NLP: VADER + TextBlob · Vectorization: TF-IDF"
)

# ── Load data ────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    csv_candidates = [
        OUTPUTS / "analyzed_music_posts.csv",
        Path("analyzed_music_posts.csv"),
    ]

    csv_path = next((p for p in csv_candidates if p.exists()), None)

    if csv_path is None:
        st.error(
            "Could not find analyzed_music_posts.csv. "
            "Run the pipeline first, or copy the file into outputs/."
        )
        st.stop()

    df = pd.read_csv(csv_path)

    if "textblob_sentiment" not in df.columns:
        with st.spinner("Computing TextBlob sentiment..."):
            df["textblob_sentiment"] = df["clean_text"].fillna("").apply(
                textblob_sentiment
            )

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["date"] = df["datetime"].dt.date

    return df


df = load_data()

# ── Overview metrics ─────────────────────────────────────────────────────────

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Posts", f"{len(df):,}")

with col2:
    agree = (df["sentiment"] == df["textblob_sentiment"]).mean()
    st.metric("VADER / TextBlob Agreement", f"{agree:.1%}")

with col3:
    top_sent = df["sentiment"].value_counts().idxmax()
    st.metric("Dominant Sentiment", top_sent.capitalize())

with col4:
    if "date" in df.columns:
        span = df["datetime"].dropna()

        if not span.empty:
            days = (span.max() - span.min()).days
            st.metric("Date Span", f"{days} days")
        else:
            st.metric("Date Span", "N/A")

# ── Sentiment distribution ───────────────────────────────────────────────────

st.subheader("Sentiment Distribution")

sentiment_counts = (
    df["sentiment"]
    .value_counts()
    .rename_axis("Sentiment")
    .reset_index(name="Posts")
)

st.bar_chart(sentiment_counts.set_index("Sentiment"))

# ── VADER vs TextBlob ────────────────────────────────────────────────────────

st.subheader("VADER vs TextBlob Comparison")

comparison = pd.crosstab(
    df["sentiment"],
    df["textblob_sentiment"],
    rownames=["VADER"],
    colnames=["TextBlob"],
)

st.dataframe(comparison, use_container_width=True)

# ── TF-IDF word analysis ─────────────────────────────────────────────────────

st.subheader("Top Words by TF-IDF Score")

st.caption(
    "TF-IDF surfaces distinctive words per sentiment class, rather than only "
    "the most frequent words overall."
)

tfidf_tab1, tfidf_tab2 = st.tabs(["Overall", "By Sentiment"])

with tfidf_tab1:
    top_overall = tfidf_top_words(df["clean_text"], n=20)

    if not top_overall.empty:
        st.bar_chart(top_overall.set_index("word")["tfidf_score"])
    else:
        st.info("No TF-IDF terms available for the current dataset.")

with tfidf_tab2:
    per_class = tfidf_by_sentiment(df, n=10)

    if per_class:
        cols = st.columns(len(per_class))

        for col, (label, frame) in zip(cols, per_class.items()):
            with col:
                st.markdown(f"**{label.capitalize()}**")

                if not frame.empty:
                    st.bar_chart(frame.set_index("word")["tfidf_score"])
                else:
                    st.info("No terms available.")
    else:
        st.info("No sentiment classes available for TF-IDF comparison.")

# ── Sentiment over time ──────────────────────────────────────────────────────

if "date" in df.columns and df["date"].notna().any():
    st.subheader("Sentiment Over Time")

    time_df = (
        df.groupby(["date", "sentiment"])
        .size()
        .reset_index(name="count")
        .pivot(index="date", columns="sentiment", values="count")
        .fillna(0)
    )

    st.line_chart(time_df)

# ── Static visualizations ────────────────────────────────────────────────────

image_specs = [
    ("sentiment_distribution.png", "Sentiment Distribution"),
    ("top_common_words.png", "Top Common Words"),
    ("sentiment_over_time.png", "Sentiment Trends Over Time"),
    ("topic_sentiment.png", "Topic Sentiment"),
    ("wordcloud.png", "Word Cloud"),
]

available = [
    (filename, caption)
    for filename, caption in image_specs
    if (OUTPUTS / filename).exists()
]

if available:
    st.subheader("Saved Visualizations")

    for i in range(0, len(available), 2):
        pair = available[i:i + 2]
        cols = st.columns(len(pair))

        for col, (filename, caption) in zip(cols, pair):
            col.image(str(OUTPUTS / filename), caption=caption)

# ── Sample posts ─────────────────────────────────────────────────────────────

st.subheader("Sample Analyzed Posts")

sentiment_filter = st.selectbox(
    "Filter by sentiment",
    ["All"] + sorted(df["sentiment"].dropna().unique().tolist()),
)

if sentiment_filter == "All":
    display_df = df
else:
    display_df = df[df["sentiment"] == sentiment_filter]

display_cols = [
    "clean_text",
    "sentiment",
    "textblob_sentiment",
]

if "date" in display_df.columns:
    display_cols = ["date"] + display_cols

available_cols = [
    col for col in display_cols
    if col in display_df.columns
]

st.dataframe(
    display_df[available_cols]
    .dropna(subset=["clean_text"])
    .head(50),
    use_container_width=True,
)