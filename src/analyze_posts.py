import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.metrics import accuracy_score

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

with open("data/music_posts.json", "r", encoding="utf-8") as file:
    posts = json.load(file)

df = pd.DataFrame(posts)
df["datetime"] = pd.to_datetime(df["time"], unit="s")


def extract_text(text):
    if pd.isna(text):
        return ""
    return str(text)


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()


def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


df["text"] = df["com"].apply(extract_text)
df["clean_text"] = df["text"].apply(clean_text)
df["sentiment"] = df["clean_text"].apply(get_sentiment)

def get_textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.05:
        return "positive"
    elif polarity < -0.05:
        return "negative"
    else:
        return "neutral"


df["textblob_sentiment"] = df["clean_text"].apply(get_textblob_sentiment)

agreement = accuracy_score(df["sentiment"], df["textblob_sentiment"])

print("\nModel Comparison:")
print(f"VADER vs TextBlob Agreement: {agreement:.2%}")

print("\nTextBlob Sentiment Summary:")
print(df["textblob_sentiment"].value_counts())

print("\nSentiment Summary:")
print(df["sentiment"].value_counts())

plt.figure()
df["sentiment"].value_counts().plot(kind="bar")
plt.title("Music Discussion Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.savefig("outputs/sentiment_distribution.png")
plt.show()

stop_words = {
    "the", "and", "is", "in", "to", "of", "a", "it", "that", "this", "for",
    "on", "with", "as", "are", "was", "be", "by", "or", "an", "at", "from",
    "i", "you", "he", "she", "they", "we", "me", "my", "your", "their",
    "but", "not", "have", "has", "had", "do", "does", "did", "so", "if",
    "just", "like", "its", "im", "dont", "get", "one", "would", "can", 
    "about", "good", "what", "more", "some", "when", "because",
    "them", "really", "even", "thats", "think", "people",
    "also", "much", "know", "make", "going", "still",
    "could", "should", "thing", "things", "want"
}

all_words = " ".join(df["clean_text"]).split()
filtered_words = [
    word for word in all_words
    if word not in stop_words and len(word) > 3
]

word_counts = Counter(filtered_words).most_common(15)

print("\nTop 15 Common Words:")
for word, count in word_counts:
    print(f"{word}: {count}")

words = [item[0] for item in word_counts]
counts = [item[1] for item in word_counts]

plt.figure()
plt.bar(words, counts)
plt.title("Top Common Words in /mu/ Music Discussions")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/top_common_words.png")
plt.show()

df.to_csv("outputs/analyzed_music_posts.csv", index=False)

print("\nFiles created:")
print("outputs/sentiment_distribution.png")
print("outputs/top_common_words.png")
print("outputs/analyzed_music_posts.csv")
print("outputs/sentiment_over_time.png")

# sentiment trends over time
sentiment_time = (
    df.groupby([
        pd.Grouper(key="datetime", freq="h"),
        "sentiment"
    ])
    .size()
    .unstack(fill_value=0)
)

plt.figure(figsize=(12, 6))

for sentiment in sentiment_time.columns:
    plt.plot(
        sentiment_time.index,
        sentiment_time[sentiment],
        label=sentiment
    )

plt.title("Sentiment Trends Over Time in /mu/")
plt.xlabel("Time")
plt.ylabel("Number of Posts")
plt.legend()

plt.tight_layout()

plt.savefig("outputs/sentiment_over_time.png")

plt.show()

print("sentiment_over_time.png")

# topic-based sentiment analysis

topic_keywords = {
    "albums": ["album", "albums"],
    "artists": ["artist", "artists", "band"],
    "songs": ["song", "songs", "track"],
    "love": ["love", "favorite", "best"],
    "criticism": ["bad", "worst", "hate", "shit"],
}

topic_sentiment = {}

for topic, keywords in topic_keywords.items():

    topic_posts = df[
        df["clean_text"].apply(
            lambda text: any(word in text for word in keywords)
        )
    ]

    sentiment_counts = (
        topic_posts["sentiment"]
        .value_counts()
        .to_dict()
    )

    topic_sentiment[topic] = sentiment_counts

topic_df = pd.DataFrame(topic_sentiment).fillna(0)

print("\nTopic Sentiment Analysis:")
print(topic_df)

topic_df.T.plot(kind="bar", figsize=(10, 6))

plt.title("Sentiment by Discussion Topic")
plt.xlabel("Topic")
plt.ylabel("Number of Posts")

plt.tight_layout()

plt.savefig("outputs/topic_sentiment.png")

plt.show()

print("outputs/topic_sentiment.png")

# word cloud visualization

text_blob = " ".join(filtered_words)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(text_blob)

plt.figure(figsize=(14, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title("Most Frequent Words in /mu/ Discussions")

plt.tight_layout()

plt.savefig("outputs/wordcloud.png")

plt.show()

print("outputs/wordcloud.png")