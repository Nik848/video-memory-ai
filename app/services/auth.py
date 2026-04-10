"""
Authentication and tenant context helpers.
"""
from fastapi import Header, HTTPException

from app.config import API_KEY, DEFAULT_USER_ID


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Enforce API key only when configured."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_user_id(x_user_id: str = Header(default=DEFAULT_USER_ID)) -> str:
    """Resolve tenant/user identifier from request header."""
    user_id = (x_user_id or DEFAULT_USER_ID).strip()
    return user_id or DEFAULT_USER_ID
