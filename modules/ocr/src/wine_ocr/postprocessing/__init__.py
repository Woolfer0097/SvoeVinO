"""Postprocessing utilities for OCR text."""

from .blocks import build_text_blocks, deduplicate_text_blocks, sort_text_blocks
from .fields import extract_candidate_fields
from .names import extract_candidate_name
from .normalization import normalize_text, tokenize_text

__all__ = [
    "build_text_blocks",
    "deduplicate_text_blocks",
    "extract_candidate_fields",
    "extract_candidate_name",
    "normalize_text",
    "sort_text_blocks",
    "tokenize_text",
]
