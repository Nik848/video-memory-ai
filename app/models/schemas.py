"""
SQLAlchemy ORM models for the Video Memory AI system.

Tables:
  - videos: metadata about each ingested video
  - chunks: text chunks extracted from videos (speech + OCR)
  - jobs: async processing job tracking
"""
from sqlalchemy import Column, String, Float, Integer, Text, DateTime, Enum as SAEnum, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import enum

from app.models.database import Base


class VideoStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkType(str, enum.Enum):
    SPEECH = "speech"
    OCR = "ocr"
    MIXED = "mixed"


class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True)  # UUID
    user_id = Column(String, nullable=False, default="public")
    url = Column(String, nullable=False)
    source_fingerprint = Column(String(255), nullable=True, index=True)
    title = Column(String, nullable=True)
    category = Column(String(255), nullable=True)
    duration = Column(Float, nullable=True)
    video_path = Column(String, nullable=True)
    audio_path = Column(String, nullable=True)
    transcript_path = Column(String, nullable=True)
    status = Column(String, default=VideoStatus.QUEUED)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    chunks = relationship("Chunk", back_populates="video", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True)  # UUID
    user_id = Column(String, nullable=False, default="public")
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    text = Column(Text, nullable=False)
    start_time = Column(Float, nullable=True)
    end_time = Column(Float, nullable=True)
    chunk_type = Column(String, default=ChunkType.SPEECH)
    cluster_id = Column(Integer, nullable=True)
    cluster_label = Column(String(255), nullable=True)
    faiss_index = Column(Integer, nullable=True)  # position in FAISS index
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    video = relationship("Video", back_populates="chunks")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)  # UUID
    user_id = Column(String, nullable=False, default="public")
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    status = Column(String, default=VideoStatus.QUEUED)
    current_step = Column(String, nullable=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    next_retry_at = Column(DateTime, nullable=True)
    dead_lettered_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


class QueryMetric(Base):
    __tablename__ = "query_metrics"

    id = Column(String, primary_key=True)  # UUID
    user_id = Column(String, nullable=False, default="public")
    query_text = Column(Text, nullable=False)
    query_intent = Column(String(100), nullable=True)
    latency_ms = Column(Float, nullable=True)
    precision_at_k = Column(Float, nullable=True)
    recall_at_k = Column(Float, nullable=True)
    answer_relevance = Column(Float, nullable=True)
    feedback = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
