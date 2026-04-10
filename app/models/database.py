"""
SQLAlchemy database setup and session management.
Uses SQLite for simplicity — can swap to Postgres later.
"""
from sqlalchemy import create_engine
from sqlalchemy import inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URL

_is_sqlite = DATABASE_URL.startswith("sqlite")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for FastAPI — yields a DB session, auto-closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called once at startup."""
    Base.metadata.create_all(bind=engine)
    _ensure_videos_category_column()
    _ensure_schema_compatibility()


def _ensure_videos_category_column():
    """Lightweight schema compatibility for legacy SQLite DBs."""
    inspector = inspect(engine)
    if "videos" not in inspector.get_table_names():
        return

    existing_columns = {c["name"] for c in inspector.get_columns("videos")}
    if "category" in existing_columns:
        return

    with engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE videos ADD COLUMN category VARCHAR(255)")
        )


def _ensure_schema_compatibility():
    """Add non-breaking columns for legacy databases."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    updates = [
        ("videos", "user_id", "ALTER TABLE videos ADD COLUMN user_id VARCHAR DEFAULT 'public'"),
        ("videos", "source_fingerprint", "ALTER TABLE videos ADD COLUMN source_fingerprint VARCHAR(255)"),
        ("chunks", "user_id", "ALTER TABLE chunks ADD COLUMN user_id VARCHAR DEFAULT 'public'"),
        ("chunks", "cluster_label", "ALTER TABLE chunks ADD COLUMN cluster_label VARCHAR(255)"),
        ("jobs", "user_id", "ALTER TABLE jobs ADD COLUMN user_id VARCHAR DEFAULT 'public'"),
        ("jobs", "attempt_count", "ALTER TABLE jobs ADD COLUMN attempt_count INTEGER DEFAULT 0"),
        ("jobs", "max_attempts", "ALTER TABLE jobs ADD COLUMN max_attempts INTEGER DEFAULT 3"),
        ("jobs", "next_retry_at", "ALTER TABLE jobs ADD COLUMN next_retry_at DATETIME"),
        ("jobs", "dead_lettered_at", "ALTER TABLE jobs ADD COLUMN dead_lettered_at DATETIME"),
    ]

    for table_name, column_name, ddl in updates:
        if table_name not in tables:
            continue
        existing_columns = {c["name"] for c in inspector.get_columns(table_name)}
        if column_name in existing_columns:
            continue
        with engine.begin() as connection:
            connection.execute(text(ddl))
