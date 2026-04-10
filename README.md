# Reely — Video Memory AI

Turn short-form videos into a searchable memory system.

Reely ingests videos from URLs, extracts speech + on-screen text, converts content into embeddings, stores vectors in FAISS, and lets you ask natural-language questions across your video library.

---

## Project Summary

This is a FastAPI-based backend (with a built-in lightweight UI) for:

- ingesting videos (sync or async with DB-backed worker queue)
- extracting knowledge from audio + OCR
- semantically indexing chunks
- querying with retrieval + optional reranking + LLM answer generation
- tracking jobs, statuses, and video-level analytics

---

## Problem Statement

Short-form videos contain useful information, but that knowledge is hard to reuse:

- content is unstructured
- search in video timelines is limited
- key ideas are locked in audio and visuals
- cross-video question answering is difficult

Reely solves this by converting videos into structured, searchable, and queryable memory.

---

## Solution Implemented

For each video URL, the ingestion pipeline performs:

1. **Download** video (`yt-dlp`)
2. **Extract audio** (`pydub` / ffmpeg)
3. **Transcribe speech** (Whisper)
4. **Extract on-screen text** via OCR (PaddleOCR preferred, EasyOCR fallback, non-fatal if it fails)
5. **Chunk content** into searchable units
6. **Normalize to English** (best-effort)
7. **Categorize video** using Ollama (`mistral`)
8. **Generate embeddings** (`sentence-transformers`)
9. **Index in FAISS** for vector search
10. **Store metadata/chunks/jobs** in SQLite (SQLAlchemy)
11. **Cluster chunks** for thematic exploration (non-fatal)

Query flow:

- embed user query
- retrieve top matches from FAISS
- optionally rerank
- optionally generate final answer with Ollama using retrieved context

---

## Tech Stack

- **Backend/API:** FastAPI, Pydantic
- **Database:** SQLAlchemy with SQLite (default) or Postgres via `DATABASE_URL`
- **Vector Search:** FAISS
- **Embeddings / Reranking:** sentence-transformers
- **ASR:** OpenAI Whisper
- **OCR:** PaddleOCR (preferred) + EasyOCR fallback + OpenCV
- **Video Download:** yt-dlp
- **Audio Processing:** pydub (+ ffmpeg)
- **LLM:** Ollama (`mistral:latest`)
- **Server:** Uvicorn

---

## Repository Structure

```text
app/
  main.py                 # FastAPI entrypoint + built-in UI
  config.py               # Central configuration
  routes/
    ingest.py             # Ingestion + job APIs
    query.py              # Semantic query/search APIs
    videos.py             # Video listing, stats, clusters
  services/
    pipeline.py           # End-to-end ingestion orchestrator
    downloader.py
    audio_extractor.py
    transcriber.py
    ocr_engine.py
    chunker.py
    normalization.py
    categorizer.py
    embedder.py
    vector_store.py
    query_engine.py
  models/
    database.py
    schemas.py
```

---

## Prerequisites

- Python 3.10+
- ffmpeg installed and available in PATH
- Ollama installed and running locally
- `mistral:latest` model pulled in Ollama

Start Ollama and pull model:

```bash
ollama serve
ollama pull mistral:latest
```

---

## Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn sqlalchemy pydantic requests numpy faiss-cpu sentence-transformers openai-whisper yt-dlp pydub easyocr opencv-python langdetect
```

> Note: Some environments may require platform-specific packages for FAISS, Whisper, or OCR dependencies.

---

## How to Run the App

Start the API server:

```bash
uvicorn app.main:app --reload
```

Then open:

- Built-in UI: `http://localhost:8000/ui`
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Ollama health: `http://localhost:8000/llm-health`

---

## Core API Endpoints

### Ingestion

- `POST /ingest/` — ingest video synchronously
- `POST /ingest/async` — enqueue async ingestion in DB-backed queue worker
- `GET /ingest/status/{job_id}` — check ingestion progress
- `POST /ingest/retry/{job_id}` — retry failed job

### Query

- `POST /query/` — semantic query + query classification + optional LLM answer
- `GET /query/search?q=...&top_k=10` — raw semantic retrieval (no LLM)

### Video Management

- `GET /videos/` — list videos
- `GET /videos/{video_id}` — full video details + chunks
- `GET /videos/stats` — aggregate stats
- `GET /videos/clusters?recompute=true` — cluster exploration

---

## Typical Usage Flow

1. Submit a video URL to `/ingest/async`
2. Poll `/ingest/status/{job_id}` until completed
3. Ask questions at `/query/`
4. Review metadata/chunks via `/videos/*`

---

## Notes

- OCR, categorization, and clustering are best-effort stages; failures there do not necessarily fail full ingestion.
- Async queue includes retry with exponential backoff and dead-letter behavior after max attempts.
- Multi-user separation is supported through `X-User-Id` header (defaults to `public`).
- Optional API key auth is enabled when `API_KEY` env var is set (`X-API-Key` header).
- You can force duplicate URL reprocessing via `POST /ingest/async` with `{ "force_reingest": true }`.
- API versioned routes are also available at `/api/v1/*`.
- Data and generated artifacts are stored locally (`db/`, `faiss_index/`, `frames/`, etc.).
- CORS is open for development by default.
