# video-memory-ai

FastAPI app for ingesting videos, indexing content, and querying across it.

## LLM model

- This project uses Ollama with the `mistral` model for:
  - query answering over ingested video content
  - automatic short-video category classification during ingestion

## Quick start

1. Start the API:
   ```bash
   uvicorn app.main:app --reload
   ```
2. Open the UI in your browser:
   - `http://localhost:8000/ui`
3. Use the UI to:
   - submit video URLs for ingestion
   - check job status / retry failed jobs
   - ask semantic questions
   - browse videos and stats

## API docs

- Swagger UI: `http://localhost:8000/docs`
