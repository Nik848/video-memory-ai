"""
Reely — AI Video Memory System
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
