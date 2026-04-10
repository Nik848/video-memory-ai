"""
Lightweight query intent classification.
"""
from typing import Dict


def classify_query(query: str) -> Dict[str, str]:
    text = (query or "").strip().lower()
    if any(k in text for k in ["cluster", "theme", "topic", "group"]):
        return {"intent": "cluster_exploration"}
    if any(k in text for k in ["compare", "difference", "versus", "vs"]):
        return {"intent": "comparison"}
    if any(k in text for k in ["summary", "summarize", "overview"]):
        return {"intent": "summary"}
    if any(k in text for k in ["when", "where", "who", "what", "how", "why"]):
        return {"intent": "fact_lookup"}
    return {"intent": "general_search"}
