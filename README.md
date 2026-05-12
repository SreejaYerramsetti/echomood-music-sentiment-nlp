# /mu/ Sentiment Analysis Crawler

Crawls 4chan's `/mu/` board, runs NLP analysis on posts, and serves results in a Streamlit dashboard. Built during Binghamton University CS 415/515 (2024).

## Architecture

```
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
```

## NLP concepts used

| Stage | Concept | File |
|---|---|---|
| Cleaning | Tokenisation, stopword removal, lemmatisation | `nlp_pipeline.py` |
| Scoring | VADER sentiment, TextBlob polarity | `nlp_pipeline.py` |
| Vectorisation | TF-IDF (replaces raw word counts) | `nlp_pipeline.py` |
| Pipeline | Faktory producer/consumer, async job queue | `crawler_worker.py` |

## Setup

### 1. Python environment

```bash
python -m venv env/dev
source env/dev/bin/activate
pip install -r requirements.txt
```

### 2. NLTK data (first run only)

```python
import nltk
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg)
```

### 3. Environment variables

Copy `.env.example` to `.env` and fill in:

```
FAKTORY_URL=tcp://:password@localhost:7419
DATABASE_URL=postgres://postgres:testpassword@localhost:5432/chan_crawler
BOARD=mu
REQUEST_DELAY=1.0
```

### 4. Postgres (TimescaleDB via Docker)

```bash
docker pull timescale/timescaledb-ha:pg16
docker run -d --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=testpassword \
  timescale/timescaledb-ha:pg16
```

### 5. Faktory (job queue)

```bash
docker run -it --name faktory \
  -v ~/projects/docker-disks/faktory-data:/var/lib/faktory/db \
  -e "FAKTORY_PASSWORD=password" \
  -p 127.0.0.1:7419:7419 \
  -p 127.0.0.1:7420:7420 \
  contribsys/faktory:latest \
  /faktory -b :7419 -w :7420
```

Faktory Web UI: http://localhost:7420

## Running the pipeline

```bash
# Terminal 1 – start worker (consumers)
python crawler_worker.py consume

# Terminal 2 – kick off a full board scrape
python crawler_worker.py produce
```

## Running the dashboard

```bash
# Put your analyzed CSV in outputs/ first
mkdir -p outputs
streamlit run app.py
```

## Project layout

```
.
├── nlp_pipeline.py      # text cleaning, sentiment, TF-IDF
├── crawler_worker.py    # Faktory jobs: scrape → NLP → Postgres
├── app.py               # Streamlit dashboard
├── requirements.txt
├── .env                 # secrets (gitignored)
└── outputs/             # generated CSVs and PNGs (gitignored)
```
