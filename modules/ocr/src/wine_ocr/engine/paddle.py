"""Lazy PaddleOCR backend wrapper.

This module deliberately does not import PaddleOCR at module import time. The
client is created only when recognition is requested.
"""

from __future__ import annotations

from importlib import import_module
from itertools import zip_longest
from typing import Any, Iterable, Iterator

from ..config import OCRConfig
from ..contracts import BoundingBox, Point, RecognizedTextBlock
from ..exceptions import OCREngineError


class PaddleOCREngine:
    """PaddleOCR text recognition backend with lazy client creation."""

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return f"paddleocr:{self.config.paddle_lang}:{self.config.device}"

    def recognize(self, image: Any, source_variant: str) -> list[RecognizedTextBlock]:
        """Run PaddleOCR on one prepared image variant."""

        try:
            numpy = import_module("numpy")
            image_array = numpy.asarray(image)
            client = self._get_client()
            raw_result = self._run_client(client, image_array)
        except Exception as exc:  # pragma: no cover - exercised with real backend
            raise OCREngineError("PaddleOCR recognition failed") from exc

        blocks: list[RecognizedTextBlock] = []
        for raw_box, text, confidence in _iter_paddle_items(raw_result):
            clean_text = str(text).strip()
            if not clean_text:
                continue
            blocks.append(
                RecognizedTextBlock(
                    text=clean_text,
                    confidence=_to_confidence(confidence),
                    bbox=_to_bounding_box(raw_box),
                    source_variant=source_variant,
                )
            )
        return blocks

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        paddleocr = import_module("paddleocr")
        paddle_cls = getattr(paddleocr, "PaddleOCR")

        kwargs: dict[str, Any] = {
            "lang": self.config.paddle_lang,
            "use_doc_orientation_classify": self.config.use_doc_orientation_classify,
            "use_doc_unwarping": self.config.use_doc_unwarping,
            "use_textline_orientation": self.config.use_textline_orientation,
        }
        if self.config.device != "auto":
            kwargs["device"] = self.config.device
        if self.config.text_det_limit_side_len is not None:
            kwargs["text_det_limit_side_len"] = self.config.text_det_limit_side_len
        if self.config.text_det_limit_type is not None:
            kwargs["text_det_limit_type"] = self.config.text_det_limit_type
        if self.config.text_det_thresh is not None:
            kwargs["text_det_thresh"] = self.config.text_det_thresh
        if self.config.text_det_box_thresh is not None:
            kwargs["text_det_box_thresh"] = self.config.text_det_box_thresh
        if self.config.text_detection_model_name is not None:
            kwargs["text_detection_model_name"] = self.config.text_detection_model_name
        if self.config.text_recognition_model_name is not None:
            kwargs["text_recognition_model_name"] = self.config.text_recognition_model_name
        try:
            return paddle_cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise OCREngineError("Cannot initialize PaddleOCR client") from exc

    def _run_client(self, client: Any, image_array: Any) -> Any:
        if hasattr(client, "predict"):
            return client.predict(image_array)
        if hasattr(client, "ocr"):
            return client.ocr(image_array)

        raise OCREngineError("PaddleOCR client exposes neither ocr() nor predict()")


def _iter_paddle_items(result: Any) -> Iterator[tuple[Any, str, Any]]:
    """Yield ``(bbox, text, confidence)`` from common PaddleOCR result shapes."""

    if result is None:
        return

    if isinstance(result, dict):
        yield from _iter_dict_result(result)
        return

    parsed = _parse_line_result(result)
    if parsed is not None:
        yield parsed
        return

    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        for item in result:
            yield from _iter_paddle_items(item)


def _iter_dict_result(result: dict[str, Any]) -> Iterator[tuple[Any, str, Any]]:
    if "text" in result:
        yield (
            result.get("bbox") or result.get("box") or result.get("points"),
            str(result["text"]),
            result.get("confidence") or result.get("score"),
        )
        return

    texts = result.get("rec_texts") or result.get("texts")
    if not texts:
        return
    scores = result.get("rec_scores") or result.get("scores") or []
    boxes = (
        result.get("rec_polys")
        or result.get("dt_polys")
        or result.get("rec_boxes")
        or result.get("boxes")
        or []
    )
    for text, score, box in zip_longest(texts, scores, boxes):
        if text is not None:
            yield box, str(text), score


def _parse_line_result(candidate: Any) -> tuple[Any, str, Any] | None:
    if not isinstance(candidate, (list, tuple)) or len(candidate) < 2:
        return None

    raw_box = candidate[0]
    recognition = candidate[1]
    if isinstance(recognition, dict) and "text" in recognition:
        return raw_box, str(recognition["text"]), recognition.get("score")
    if isinstance(recognition, (list, tuple)) and recognition:
        text = recognition[0]
        score = recognition[1] if len(recognition) > 1 else None
        if isinstance(text, str):
            return raw_box, text, score
    return None


def _to_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def _to_bounding_box(raw_box: Any) -> BoundingBox | None:
    if raw_box is None:
        return None
    if hasattr(raw_box, "tolist"):
        raw_box = raw_box.tolist()

    try:
        if _looks_like_xyxy(raw_box):
            left, top, right, bottom = (float(value) for value in raw_box)
            return BoundingBox(
                (
                    Point(left, top),
                    Point(right, top),
                    Point(right, bottom),
                    Point(left, bottom),
                )
            )

        points = tuple(Point(float(point[0]), float(point[1])) for point in raw_box)
    except (TypeError, ValueError, IndexError):
        return None

    return BoundingBox(points) if points else None


def _looks_like_xyxy(raw_box: Any) -> bool:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return False
    return all(isinstance(value, (int, float)) for value in raw_box)
