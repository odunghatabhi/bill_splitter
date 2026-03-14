from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env in local dev (safe even if file doesn't exist)
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Backend
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Gemini
    default_model: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    # Limits
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))


settings = Settings()
