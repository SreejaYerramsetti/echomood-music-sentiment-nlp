````md
# EchoMood: A Scalable NLP Pipeline for Music Discussion Sentiment Analysis

EchoMood crawls 4chan's `/mu/` music board, runs NLP-based sentiment analysis on discussion posts, and serves the results through an interactive Streamlit dashboard. Built for Binghamton University CS 415/515 (2026).

## Architecture

```text
scrape_board job
      │
      ▼
Faktory queue ("default")
      │
      ▼
scrape_thread jobs  ──►  Faktory queue ("threads")
      │
      ▼
nlp_pipeline.py          ← clean_text, VADER, TextBlob, TF-IDF
      │
      ▼
Postgres (TimescaleDB)
      │
      ▼
app.py  (Streamlit dashboard)
````

## NLP concepts used

| Stage         | Concept                                       | File                |
| ------------- | --------------------------------------------- | ------------------- |
| Cleaning      | Tokenization, stopword removal, lemmatization | `nlp_pipeline.py`   |
| Scoring       | VADER sentiment, TextBlob polarity            | `nlp_pipeline.py`   |
| Vectorization | TF-IDF feature extraction                     | `nlp_pipeline.py`   |
| Pipeline      | Faktory producer/consumer, async job queue    | `crawler_worker.py` |

## Data collection

Posts are fetched live from the [4chan public JSON API](https://github.com/4chan/4chan-API). No authentication is required.

* Board catalog: `https://a.4cdn.org/mu/catalog.json`
* Per-thread posts: `https://a.4cdn.org/mu/thread/{thread_id}.json`

Raw JSON is processed in memory and written to Postgres. A 1-second courtesy delay between requests helps avoid rate-limiting.

## Setup

### 1. Python environment

```bash
python -m venv env/dev
source env/dev/bin/activate
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
.\env\dev\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. NLTK data

```python
import nltk

for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg)
```

### 3. Environment variables

Create a `.env` file:

```env
FAKTORY_URL=tcp://:password@localhost:7419
DATABASE_URL=postgres://postgres:testpassword@localhost:5432/chan_crawler
BOARD=mu
REQUEST_DELAY=1.0
```

### 4. Postgres / TimescaleDB

```bash
docker pull timescale/timescaledb-ha:pg16

docker run -d --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=testpassword \
  timescale/timescaledb-ha:pg16
```

### 5. Faktory job queue

```bash
docker run -it --name faktory \
  -v ~/projects/docker-disks/faktory-data:/var/lib/faktory/db \
  -e "FAKTORY_PASSWORD=password" \
  -p 127.0.0.1:7419:7419 \
  -p 127.0.0.1:7420:7420 \
  contribsys/faktory:latest \
  /faktory -b :7419 -w :7420
```

Faktory Web UI:

```text
http://localhost:7420
```

## Running the pipeline

Start the worker:

```bash
python crawler_worker.py consume
```

In a second terminal, start a board scrape:

```bash
python crawler_worker.py produce
```

## Running the dashboard

The dashboard reads from the `outputs/` folder.

```bash
mkdir -p outputs
cp analyzed_music_posts.csv outputs/
cp sentiment_distribution.png outputs/
cp top_common_words.png outputs/
streamlit run app.py
```

For Windows PowerShell:

```powershell
mkdir outputs
Copy-Item analyzed_music_posts.csv outputs/
Copy-Item sentiment_distribution.png outputs/
Copy-Item top_common_words.png outputs/
streamlit run app.py
```

## Project layout

```text
.
├── app.py                         # Streamlit dashboard
├── crawler_worker.py              # Faktory jobs: scrape → NLP → Postgres
├── nlp_pipeline.py                # text cleaning, sentiment, TF-IDF
├── requirements.txt
├── README.md
├── .env                           # local secrets, gitignored
├── analyzed_music_posts.csv       # sample analyzed dataset
├── sentiment_distribution.png     # pre-generated visualization
├── top_common_words.png           # pre-generated visualization
├── data/                          # optional raw/intermediate data
├── outputs/                       # dashboard-ready outputs
└── reports/                       # project report files
```

## Project summary

EchoMood demonstrates a complete NLP data pipeline for online music discussion analysis. It combines live data collection, asynchronous job processing, text preprocessing, dual-model sentiment scoring, TF-IDF term analysis, persistent database storage, and dashboard-based visualization.

The system is designed to be modular: `crawler_worker.py` handles data ingestion and job orchestration, `nlp_pipeline.py` contains reusable NLP functions, and `app.py` presents the results through a user-facing dashboard.

```

