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
    url = Column(String, nullable=False)
    title = Column(String, nullable=True)
    category = Column(String, nullable=True)
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
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    text = Column(Text, nullable=False)
    start_time = Column(Float, nullable=True)
    end_time = Column(Float, nullable=True)
    chunk_type = Column(String, default=ChunkType.SPEECH)
    cluster_id = Column(Integer, nullable=True)
    faiss_index = Column(Integer, nullable=True)  # position in FAISS index
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    video = relationship("Video", back_populates="chunks")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)  # UUID
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    status = Column(String, default=VideoStatus.QUEUED)
    current_step = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
