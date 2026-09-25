"""Filename helpers for generated OCR reports."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4


def safe_report_filename(
    original_filename: str | None,
    *,
    suffix: str | None = None,
) -> str:
    """Return a filesystem-safe TXT report name for an uploaded file."""

    stem = Path(original_filename or "upload").stem
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", stem).strip(".-")
    if not normalized:
        normalized = "upload"
    unique_suffix = suffix if suffix is not None else uuid4().hex
    return f"{normalized}-{unique_suffix}.txt"
