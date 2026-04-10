"""
Reely — AI Video Memory System
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging

from app.models.database import init_db
from app.routes import ingest, query, videos

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Reely — AI Video Memory System",
    description="Convert short-form video content into a searchable knowledge base",
    version="1.0.0"
)

# CORS (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(videos.router)


@app.on_event("startup")
def startup():
    """Initialize database tables on startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Reely API is ready!")


@app.get("/")
def root():
    return {
        "service": "Reely — AI Video Memory System",
        "version": "1.0.0",
        "endpoints": {
            "ui": "/ui",
            "ingest": "/ingest/ (POST)",
            "ingest_async": "/ingest/async (POST)",
            "ingest_status": "/ingest/status/{job_id} (GET)",
            "ingest_retry": "/ingest/retry/{job_id} (POST)",
            "query": "/query/ (POST)",
            "search": "/query/search?q=... (GET)",
            "videos": "/videos/ (GET)",
            "video_detail": "/videos/{id} (GET)",
            "video_stats": "/videos/stats (GET)",
            "clusters": "/videos/clusters?recompute=true (GET)",
            "docs": "/docs"
        }
    }


@app.get("/ui", response_class=HTMLResponse)
def ui():
    """Simple built-in web UI for ingestion, querying, and video management."""
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Reely UI</title>
  <style>
    :root {
      --bg: #0b1020;
      --card: #121a2f;
      --muted: #96a2c0;
      --text: #e8edff;
      --accent: #6ea8ff;
      --accent-2: #4de2c5;
      --border: #243055;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: radial-gradient(circle at top right, #1a2350, var(--bg));
      color: var(--text);
      padding: 20px;
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }
    .card {
      background: #111933;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 26px rgba(0,0,0,.25);
    }
    h1 {
      margin: 0 0 12px;
      font-size: 1.6rem;
      grid-column: 1 / -1;
    }
    h2 { margin: 0 0 10px; font-size: 1rem; }
    p { margin: 6px 0 10px; color: var(--muted); font-size: .92rem; }
    input, textarea {
      width: 100%;
      background: #0d1430;
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 10px;
      padding: 10px;
      margin: 6px 0;
    }
    textarea { min-height: 90px; resize: vertical; }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 12px;
      font-weight: 600;
      color: #06122b;
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
    }
    button.secondary {
      background: #1f2a4a;
      color: #d9e4ff;
      border: 1px solid var(--border);
    }
    pre {
      background: #0d1430;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      margin-top: 10px;
      max-height: 320px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: .83rem;
      line-height: 1.35;
    }
    .muted { color: var(--muted); font-size: .85rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Reely — Video Memory UI</h1>

    <section class="card">
      <h2>Ingest Video (Async)</h2>
      <p>Submit a video URL and track progress with job ID.</p>
      <input id="ingestUrl" placeholder="https://..." />
      <button onclick="ingestAsync()">Submit for Processing</button>
      <pre id="ingestOut"></pre>
    </section>

    <section class="card">
      <h2>Track Job</h2>
      <input id="jobId" placeholder="job_id" />
      <div class="row">
        <button onclick="jobStatus()">Check Status</button>
        <button class="secondary" onclick="retryJob()">Retry Failed Job</button>
      </div>
      <pre id="jobOut"></pre>
    </section>

    <section class="card">
      <h2>Ask Questions</h2>
      <textarea id="question" placeholder="Ask anything across all ingested videos..."></textarea>
      <div class="row">
        <input id="topK" type="number" value="10" min="1" />
        <button onclick="ask()">Run Query</button>
      </div>
      <pre id="queryOut"></pre>
    </section>

    <section class="card">
      <h2>Semantic Search (Raw Chunks)</h2>
      <input id="searchQ" placeholder="search text..." />
      <button onclick="search()">Search</button>
      <pre id="searchOut"></pre>
    </section>

    <section class="card">
      <h2>Videos & Details</h2>
      <div class="row">
        <button onclick="listVideos()">Refresh Video List</button>
        <button class="secondary" onclick="stats()">Load Stats</button>
      </div>
      <p class="muted">Tip: click any video ID from list and paste below.</p>
      <input id="videoId" placeholder="video_id for details" />
      <button onclick="videoDetails()">Get Video Details</button>
      <pre id="videosOut"></pre>
    </section>
  </div>

  <script>
    async function api(method, path, body = null) {
      const opts = { method, headers: { "Content-Type": "application/json" } };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch(path, opts);
      const text = await res.text();
      let data;
      try { data = JSON.parse(text); } catch { data = text; }
      if (!res.ok) throw new Error(typeof data === "string" ? data : JSON.stringify(data, null, 2));
      return data;
    }

    function print(id, value) {
      document.getElementById(id).textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    }

    async function ingestAsync() {
      const url = document.getElementById("ingestUrl").value.trim();
      if (!url) return print("ingestOut", "Enter a video URL.");
      try {
        const data = await api("POST", "/ingest/async", { url });
        print("ingestOut", data);
        if (data.job_id) document.getElementById("jobId").value = data.job_id;
        if (data.video_id) document.getElementById("videoId").value = data.video_id;
      } catch (e) { print("ingestOut", e.message); }
    }

    async function jobStatus() {
      const jobId = document.getElementById("jobId").value.trim();
      if (!jobId) return print("jobOut", "Enter a job_id.");
      try { print("jobOut", await api("GET", `/ingest/status/${jobId}`)); }
      catch (e) { print("jobOut", e.message); }
    }

    async function retryJob() {
      const jobId = document.getElementById("jobId").value.trim();
      if (!jobId) return print("jobOut", "Enter a job_id.");
      try { print("jobOut", await api("POST", `/ingest/retry/${jobId}`)); }
      catch (e) { print("jobOut", e.message); }
    }

    async function ask() {
      const question = document.getElementById("question").value.trim();
      const topK = parseInt(document.getElementById("topK").value || "10", 10);
      if (!question) return print("queryOut", "Enter a question.");
      try {
        print("queryOut", await api("POST", "/query/", {
          question,
          top_k: Number.isFinite(topK) ? topK : 10,
          use_reranker: true,
          use_llm: true
        }));
      } catch (e) { print("queryOut", e.message); }
    }

    async function search() {
      const q = document.getElementById("searchQ").value.trim();
      if (!q) return print("searchOut", "Enter search text.");
      try { print("searchOut", await api("GET", `/query/search?q=${encodeURIComponent(q)}&top_k=10`)); }
      catch (e) { print("searchOut", e.message); }
    }

    async function listVideos() {
      try { print("videosOut", await api("GET", "/videos/")); }
      catch (e) { print("videosOut", e.message); }
    }

    async function videoDetails() {
      const videoId = document.getElementById("videoId").value.trim();
      if (!videoId) return print("videosOut", "Enter a video_id.");
      try { print("videosOut", await api("GET", `/videos/${videoId}`)); }
      catch (e) { print("videosOut", e.message); }
    }

    async function stats() {
      try { print("videosOut", await api("GET", "/videos/stats/")); }
      catch (e) { print("videosOut", e.message); }
    }
  </script>
</body>
</html>
    """


@app.get("/health")
def health():
    """Health check endpoint."""
    from app.services.vector_store import get_total_vectors
    from app.models.database import SessionLocal
    from app.models.schemas import Video, Chunk

    db = SessionLocal()
    try:
        return {
            "status": "healthy",
            "videos": db.query(Video).count(),
            "chunks": db.query(Chunk).count(),
            "vectors": get_total_vectors()
        }
    finally:
        db.close()
