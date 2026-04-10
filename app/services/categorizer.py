"""
Video Categorization Service
Classifies a video into a single category using Ollama LLM.
"""
import logging
import re
from typing import List, Optional

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, VIDEO_CATEGORIES

logger = logging.getLogger(__name__)

MAX_CHUNKS_FOR_CATEGORIZATION = 20
MAX_CHUNK_TEXT_LENGTH = 300
CATEGORIZATION_TIMEOUT_SECONDS = 30


def categorize_video(title: Optional[str], chunk_texts: List[str]) -> Optional[str]:
    """
    Classify video content into one category label.
    Returns None if categorization is unavailable.
    """
    if not chunk_texts:
        return None

    sampled_chunks = _sample_chunks_evenly(
        chunk_texts,
        max_items=MAX_CHUNKS_FOR_CATEGORIZATION,
    )
    context = "\n".join(
        f"- {_truncate_text(text, MAX_CHUNK_TEXT_LENGTH)}"
        for text in sampled_chunks
        if text
    )
    if not context.strip():
        return None

    categories_csv = ", ".join(VIDEO_CATEGORIES)
    prompt = f"""Classify this video into exactly one category from this list:
{categories_csv}

Video title: {title or "Unknown"}
Video content:
{context}

Rules:
- Return only one category from the list.
- Output just the category text with no explanation.
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=CATEGORIZATION_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            logger.warning(
                "Ollama categorization returned %s: %s",
                response.status_code,
                response.text[:500],
            )
            return None

        raw_output = response.json().get("response", "")
        category = _normalize_category(raw_output)
        if category:
            logger.info(f"Video categorized as '{category}'")
            return category

        logger.warning(f"Unrecognized category output: {raw_output!r}")
        return None
    except requests.ConnectionError:
        logger.warning("Cannot connect to Ollama for categorization")
        return None
    except Exception as e:
        logger.warning(f"Video categorization failed: {e}")
        return None


def _normalize_category(raw_output: str) -> Optional[str]:
    if not raw_output:
        return None

    cleaned = raw_output.strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s-]", "", cleaned)
    cleaned = cleaned.replace("-", " ").strip()

    categories_normalized = {c.replace("-", " ").lower(): c for c in VIDEO_CATEGORIES}
    if cleaned in categories_normalized:
        return categories_normalized[cleaned]

    matches = []
    for normalized, canonical in categories_normalized.items():
        pattern = rf"\b{re.escape(normalized)}\b"
        if re.search(pattern, cleaned):
            matches.append(canonical)

    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]

    return None


def _sample_chunks_evenly(chunk_texts: List[str], max_items: int) -> List[str]:
    if not chunk_texts:
        return []
    if max_items <= 1:
        return [chunk_texts[0]]
    if len(chunk_texts) <= max_items:
        return chunk_texts

    step = (len(chunk_texts) - 1) / (max_items - 1)
    sampled = []
    for i in range(max_items):
        index = min(round(i * step), len(chunk_texts) - 1)
        sampled.append(chunk_texts[index])
    return sampled


def _truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text

    truncated = text[:max_length].strip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0].strip() or truncated
    return f"{truncated}..."
