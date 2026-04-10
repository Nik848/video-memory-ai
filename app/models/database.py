"""
SQLAlchemy database setup and session management.
Uses SQLite for simplicity — can swap to Postgres later.
"""
from sqlalchemy import create_engine
from sqlalchemy import inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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


def _ensure_videos_category_column():
    """Lightweight schema compatibility for legacy SQLite DBs."""
    inspector = inspect(engine)
    if "videos" not in inspector.get_table_names():
        return

    existing_columns = {c["name"] for c in inspector.get_columns("videos")}
    if "category" in existing_columns:
        return

    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE videos ADD COLUMN category VARCHAR"))
