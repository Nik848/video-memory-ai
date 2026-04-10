"""
Multilingual normalization service.
Detects language and translates non-English chunks to English before embedding.
"""
import logging
from typing import List, Dict

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

TRANSLATION_MIN_PREDICT_TOKENS = 256
TRANSLATION_MAX_PREDICT_TOKENS = 2048
TRANSLATION_TOKENS_PER_WORD = 3

try:
    from langdetect import detect as _detect_langdetect
    _LANGDETECT_AVAILABLE = True
except Exception:
    _detect_langdetect = None
    _LANGDETECT_AVAILABLE = False


def _detect_language(text: str) -> str:
    """Best-effort language detection with safe fallback."""
    if not text or not text.strip():
        return "unknown"

    if _LANGDETECT_AVAILABLE:
        try:
            return _detect_langdetect(text)
        except Exception:
            pass

    ascii_chars = sum(1 for c in text if c.isascii())
    ratio = ascii_chars / max(len(text), 1)
    return "en" if ratio > 0.95 else "unknown"


def _translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English via Ollama with graceful fallback."""
    if source_lang in {"en", "unknown"}:
        return text

    prompt = (
        "Translate the following text to English.\n"
        "Return only the translated text with no extra commentary.\n\n"
        f"Source language: {source_lang}\n"
        f"Text: {text}"
    )
    max_predict = min(
        TRANSLATION_MAX_PREDICT_TOKENS,
        max(
            TRANSLATION_MIN_PREDICT_TOKENS,
            len(text.split()) * TRANSLATION_TOKENS_PER_WORD
        )
    )

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": max_predict}
            },
            timeout=20
        )
        if response.status_code == 200:
            translated = response.json().get("response", "").strip()
            return translated or text
    except Exception as e:
        logger.warning(f"Translation fallback to original text: {e}")

    return text


def normalize_chunks_to_english(chunks: List[Dict]) -> List[Dict]:
    """
    Normalize chunk text to English before embedding.
    Returns a new list and keeps chunk schema unchanged.
    """
    normalized = []
    translated_count = 0

    for chunk in chunks:
        current = dict(chunk)
        source_text = current.get("text", "") or ""
        language = _detect_language(source_text)
        normalized_text = _translate_to_english(source_text, language).strip()

        if normalized_text and normalized_text != source_text:
            translated_count += 1

        current["text"] = normalized_text or source_text
        normalized.append(current)

    logger.info(
        "Normalization complete: %s chunks processed, %s translated",
        len(normalized),
        translated_count,
    )
    return normalized
