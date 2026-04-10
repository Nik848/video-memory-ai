"""
Centralized configuration for the Video Memory AI system.
All paths, model names, and service settings in one place.
"""
import os

# ── Base Directories ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")
CHUNK_DIR = os.path.join(BASE_DIR, "chunks")
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
DB_DIR = os.path.join(BASE_DIR, "db")

# Create all directories
for d in [DOWNLOAD_DIR, AUDIO_DIR, TRANSCRIPT_DIR, CHUNK_DIR,
          EMBEDDING_DIR, FAISS_DIR, FRAMES_DIR, DB_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Database ──────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(DB_DIR, 'reely.db')}"
)

# ── Models ────────────────────────────────────────────────────────
WHISPER_MODEL = "base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM (Ollama) ─────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "mistral:latest"
VIDEO_CATEGORIES = [
    "education",
    "technology",
    "business",
    "finance",
    "entertainment",
    "sports",
    "health",
    "lifestyle",
    "travel",
    "food",
    "fashion",
    "news",
    "gaming",
    "music",
    "other",
]

# ── Processing Settings ──────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1  # mono
MAX_CHUNK_WORDS = 120

# ── OCR Settings ──────────────────────────────────────────────────
OCR_FRAME_INTERVAL = 2  # extract frame every N seconds
OCR_LANGUAGES = ["en"]

# ── Search Settings ───────────────────────────────────────────────
TOP_K_RESULTS = 10
TOP_N_RERANKED = 5

# ── Ingestion Queue / Retry Settings ──────────────────────────────
QUEUE_WORKER_POLL_SECONDS = float(os.getenv("QUEUE_WORKER_POLL_SECONDS", "1.0"))
JOB_MAX_ATTEMPTS = int(os.getenv("JOB_MAX_ATTEMPTS", "3"))
JOB_BASE_BACKOFF_SECONDS = int(os.getenv("JOB_BASE_BACKOFF_SECONDS", "10"))
JOB_MAX_BACKOFF_SECONDS = int(os.getenv("JOB_MAX_BACKOFF_SECONDS", "300"))

# ── API Hardening ──────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "")
DEFAULT_USER_ID = "public"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

# ── Cookies (for yt-dlp) ─────────────────────────────────────────
COOKIES_PATH = os.path.join(BASE_DIR, "app", "services", "cookies.txt")
