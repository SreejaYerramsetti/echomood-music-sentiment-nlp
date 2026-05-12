"""
crawler_worker.py
-----------------
Faktory-powered asynchronous pipeline for EchoMood:
A Scalable NLP Pipeline for Music Discussion Sentiment Analysis.

This module scrapes posts from 4chan's /mu/ board, processes them through
the NLP pipeline, and stores analyzed results in PostgreSQL / TimescaleDB.

Architecture
------------
Producer  →  Faktory Queue  →  Consumer Workers

scrape_board job
    → fetches the /mu/ catalog
    → pushes one scrape_thread job per thread ID

scrape_thread job
    → fetches all posts from a thread
    → cleans and analyzes text
    → inserts processed records into Postgres

Run producer:
    python crawler_worker.py produce

Run consumer:
    python crawler_worker.py consume

MLOps Note
-----------
Each job is idempotent. Re-processing the same thread is safe because
database inserts use:

    INSERT ... ON CONFLICT DO NOTHING
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone

import psycopg2
import requests
from dotenv import load_dotenv
from pyfaktory import Client, Consumer, Job, Producer

from nlp_pipeline import (
    clean_text,
    vader_sentiment,
    vader_score,
    textblob_sentiment,
)

load_dotenv()

# ── Logging configuration ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

log = logging.getLogger("crawler")

# ── Environment configuration ───────────────────────────────────────────────

FAKTORY_URL = os.getenv(
    "FAKTORY_URL",
    "tcp://:password@localhost:7419"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgres://postgres:testpassword@localhost:5432/chan_crawler"
)

BOARD = os.getenv("BOARD", "mu")

# Courtesy delay between API requests
DELAY_SEC = float(os.getenv("REQUEST_DELAY", "1.0"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (educational research crawler)"
}

# ── Database helpers ────────────────────────────────────────────────────────

def get_conn():
    """Create and return a PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL)


def ensure_schema():
    """Create the database table if it does not already exist."""

    ddl = """
    CREATE TABLE IF NOT EXISTS mu_posts (
        no            BIGINT PRIMARY KEY,
        thread_id     BIGINT,
        board         TEXT,
        datetime      TIMESTAMPTZ,
        raw_text      TEXT,
        clean_text    TEXT,
        sentiment     TEXT,
        vader_score   REAL,
        tb_sentiment  TEXT,
        sub           TEXT,
        replies       INT,
        images        INT,
        inserted_at   TIMESTAMPTZ DEFAULT NOW()
    );
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)

        conn.commit()

    log.info("Database schema verified")

# ── 4chan API helpers ───────────────────────────────────────────────────────

def fetch_catalog(board: str) -> list[int]:
    """
    Fetch the /mu/ catalog and return all active thread IDs.
    """

    url = f"https://a.4cdn.org/{board}/catalog.json"

    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    thread_ids = []

    for page in resp.json():
        for thread in page.get("threads", []):
            thread_ids.append(thread["no"])

    return thread_ids


def fetch_thread(board: str, thread_id: int) -> list[dict]:
    """
    Fetch all posts from a thread as raw dictionaries.
    """

    url = f"https://a.4cdn.org/{board}/thread/{thread_id}.json"

    resp = requests.get(url, headers=HEADERS, timeout=15)

    if resp.status_code == 404:
        log.warning("Thread %s not found or deleted", thread_id)
        return []

    resp.raise_for_status()

    return resp.json().get("posts", [])

# ── NLP processing ──────────────────────────────────────────────────────────

def process_post(post: dict, thread_id: int, board: str) -> dict:
    """
    Apply the complete NLP pipeline to a raw 4chan post.

    Pipeline stages:
        • HTML cleaning
        • Tokenization
        • Stopword removal
        • Lemmatization
        • VADER sentiment scoring
        • TextBlob polarity analysis
    """

    raw = post.get("com", "")
    cleaned = clean_text(raw)

    ts = post.get("time")

    dt = (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        if ts else None
    )

    return {
        "no": post["no"],
        "thread_id": thread_id,
        "board": board,
        "datetime": dt,
        "raw_text": raw,
        "clean_text": cleaned,
        "sentiment": vader_sentiment(cleaned),
        "vader_score": vader_score(cleaned),
        "tb_sentiment": textblob_sentiment(cleaned),
        "sub": post.get("sub", ""),
        "replies": int(post.get("replies", 0) or 0),
        "images": int(post.get("images", 0) or 0),
    }


def insert_posts(posts: list[dict]):
    """
    Bulk insert processed posts into PostgreSQL.

    Duplicate posts are skipped safely using:
        ON CONFLICT DO NOTHING
    """

    if not posts:
        return

    sql = """
    INSERT INTO mu_posts
        (
            no,
            thread_id,
            board,
            datetime,
            raw_text,
            clean_text,
            sentiment,
            vader_score,
            tb_sentiment,
            sub,
            replies,
            images
        )
    VALUES
        (
            %(no)s,
            %(thread_id)s,
            %(board)s,
            %(datetime)s,
            %(raw_text)s,
            %(clean_text)s,
            %(sentiment)s,
            %(vader_score)s,
            %(tb_sentiment)s,
            %(sub)s,
            %(replies)s,
            %(images)s
        )
    ON CONFLICT (no) DO NOTHING;
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, posts)

        conn.commit()

    log.info("Inserted %d posts", len(posts))

# ── Faktory job handlers ────────────────────────────────────────────────────

def job_scrape_board(board: str):
    """
    Pipeline entry point.

    Fetch the catalog and enqueue one scrape_thread job
    for every discovered thread ID.
    """

    log.info("Scraping catalog for /%s/", board)

    thread_ids = fetch_catalog(board)

    log.info("Found %d active threads", len(thread_ids))

    with Client(faktory_url=FAKTORY_URL, role="producer") as client:

        producer = Producer(client=client)

        jobs = [
            Job(
                jobtype="scrape_thread",
                args=(board, tid),
                queue="threads",
            )
            for tid in thread_ids
        ]

        producer.push_bulk(jobs)

    log.info("Enqueued %d scrape_thread jobs", len(jobs))


def job_scrape_thread(board: str, thread_id: int):
    """
    Fetch all posts in a thread, run NLP analysis,
    and store processed results.
    """

    log.info("Processing thread %s/%s", board, thread_id)

    # Courtesy delay for API friendliness
    time.sleep(DELAY_SEC)

    raw_posts = fetch_thread(board, thread_id)

    if not raw_posts:
        return

    processed = [
        process_post(p, thread_id, board)
        for p in raw_posts
    ]

    insert_posts(processed)

# ── Main entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":

    mode = sys.argv[1] if len(sys.argv) > 1 else "consume"

    ensure_schema()

    if mode == "produce":

        with Client(
            faktory_url=FAKTORY_URL,
            role="producer"
        ) as client:

            producer = Producer(client=client)

            job = Job(
                jobtype="scrape_board",
                args=(BOARD,),
                queue="default",
            )

            producer.push(job)

        log.info("Pushed scrape_board job for /%s/", BOARD)

    elif mode == "consume":

        log.info("Starting Faktory consumer workers (concurrency=4)")

        with Client(
            faktory_url=FAKTORY_URL,
            role="consumer"
        ) as client:

            consumer = Consumer(
                client=client,
                queues=["default", "threads"],
                concurrency=4,
            )

            consumer.register(
                "scrape_board",
                job_scrape_board,
            )

            consumer.register(
                "scrape_thread",
                job_scrape_thread,
            )

            consumer.run()

    else:
        print("Usage: python crawler_worker.py [produce|consume]")
        sys.exit(1)