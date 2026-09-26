"""Multilingual E5 query embeddings for text recognized on a wine label."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

DEFAULT_TEXT_MODEL = "intfloat/multilingual-e5-base"
MAX_MODEL_TOKENS = 512
QUERY_PREFIX = "query: "


class TextEmbeddingError(RuntimeError):
    """The text model could not produce a usable query vector."""


@dataclass(frozen=True)
class TextEmbedding:
    """One query vector, including the model identity required by the catalog."""

    values: tuple[float, ...]
    model_name: str
    chunk_count: int = 1

    @property
    def dimension(self) -> int:
        return len(self.values)


class TextEmbedder(Protocol):
    def embed(self, text: str) -> TextEmbedding:
        """Include the entire OCR text in a query embedding."""


def split_query_text(text: str, tokenizer: Any, *, max_tokens: int = MAX_MODEL_TOKENS) -> list[str]:
    """Split long OCR text on whitespace so no words are silently truncated."""

    words = text.split()
    if not words:
        return []
    if len(tokenizer.encode(QUERY_PREFIX + text, add_special_tokens=True)) <= max_tokens:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join((*current, word))
        if len(tokenizer.encode(QUERY_PREFIX + candidate, add_special_tokens=True)) <= max_tokens:
            current.append(word)
            continue
        if not current:
            raise TextEmbeddingError("One OCR word exceeds the text model token limit")
        chunks.append(" ".join(current))
        current = [word]
        if len(tokenizer.encode(QUERY_PREFIX + word, add_special_tokens=True)) > max_tokens:
            raise TextEmbeddingError("One OCR word exceeds the text model token limit")
    if current:
        chunks.append(" ".join(current))
    return chunks


class E5TextEmbedder:
    """Lazy, reusable E5 encoder compatible with catalog passage embeddings."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = (model_name or os.getenv("TEXT_EMBEDDING_MODEL_NAME") or DEFAULT_TEXT_MODEL).strip()
        self.requested_device = (device or os.getenv("TEXT_EMBEDDING_DEVICE") or "auto").strip()
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._lock = Lock()

    def embed(self, text: str) -> TextEmbedding:
        """Mean-pool E5 token vectors and L2-normalize the complete OCR query."""

        if not text.strip():
            raise TextEmbeddingError("OCR text is empty")
        with self._lock:
            self._load_model()
            try:
                chunks = split_query_text(text, self._tokenizer)
                tokens = self._tokenizer(
                    [QUERY_PREFIX + chunk for chunk in chunks],
                    max_length=MAX_MODEL_TOKENS,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                ).to(self._device)
                with self._torch.inference_mode():
                    hidden = self._model(**tokens).last_hidden_state
                    mask = tokens["attention_mask"].unsqueeze(-1).bool()
                    pooled = hidden.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1)
                    vectors = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                    if len(chunks) == 1:
                        combined = vectors[0]
                    else:
                        weights = self._torch.tensor(
                            [len(chunk.split()) for chunk in chunks],
                            dtype=vectors.dtype,
                            device=self._device,
                        )
                        combined = self._torch.nn.functional.normalize(
                            (vectors * weights.unsqueeze(1)).sum(dim=0), p=2, dim=0
                        )
                values = tuple(float(value) for value in combined.float().cpu().tolist())
            except TextEmbeddingError:
                raise
            except Exception as exc:
                raise TextEmbeddingError("Cannot embed OCR text with the E5 model") from exc

        if (
            not values
            or not all(math.isfinite(value) for value in values)
            or not any(value != 0.0 for value in values)
        ):
            raise TextEmbeddingError("Text model returned an invalid vector")
        return TextEmbedding(values=values, model_name=self.model_name, chunk_count=len(chunks))

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            requested = self.requested_device.strip().lower()
            if requested not in {"auto", "cpu", "cuda"}:
                raise TextEmbeddingError("TEXT_EMBEDDING_DEVICE must be auto, cpu, or cuda")
            if requested == "cuda" and not torch.cuda.is_available():
                raise TextEmbeddingError("CUDA was requested for text embeddings but is unavailable")
            device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
            if device == "auto":
                device = "cpu"
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name).to(device).eval()
        except TextEmbeddingError:
            raise
        except Exception as exc:
            raise TextEmbeddingError("Cannot load the E5 text embedding model") from exc

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
