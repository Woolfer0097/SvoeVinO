"""Long-lived OCR runtime used by service entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from ..config import OCRConfig
from ..contracts import OCRResult
from ..engine.base import OCREngine
from ..engine.factory import create_engine
from .service import run_ocr_from_bytes


@dataclass(frozen=True)
class OCREngineKey:
    """Configuration fields that affect OCR backend construction."""

    engine: str
    device: str
    paddle_lang: str
    use_doc_orientation_classify: bool
    use_doc_unwarping: bool
    use_textline_orientation: bool
    text_det_limit_side_len: int | None
    text_det_limit_type: str | None
    text_det_thresh: float | None
    text_det_box_thresh: float | None
    text_detection_model_name: str | None
    text_recognition_model_name: str | None

    @classmethod
    def from_config(cls, config: OCRConfig) -> "OCREngineKey":
        return cls(
            engine=config.engine.strip().lower(),
            device=config.device.strip().lower(),
            paddle_lang=config.paddle_lang.strip().lower(),
            use_doc_orientation_classify=config.use_doc_orientation_classify,
            use_doc_unwarping=config.use_doc_unwarping,
            use_textline_orientation=config.use_textline_orientation,
            text_det_limit_side_len=config.text_det_limit_side_len,
            text_det_limit_type=config.text_det_limit_type,
            text_det_thresh=config.text_det_thresh,
            text_det_box_thresh=config.text_det_box_thresh,
            text_detection_model_name=config.text_detection_model_name,
            text_recognition_model_name=config.text_recognition_model_name,
        )


class CachedOCRRuntime:
    """Lazily create and reuse an OCR engine between HTTP requests."""

    def __init__(self) -> None:
        self._engine: OCREngine | None = None
        self._engine_key: OCREngineKey | None = None
        self._engine_lock = Lock()
        self._recognition_lock = Lock()

    def run_ocr_from_bytes(self, image_bytes: bytes, config: OCRConfig) -> OCRResult:
        """Run OCR with a cached backend instance."""

        engine = self.get_engine(config)
        with self._recognition_lock:
            return run_ocr_from_bytes(image_bytes, config=config, engine=engine)

    def get_engine(self, config: OCRConfig) -> OCREngine:
        """Return a cached engine, recreating it when backend settings change."""

        requested_key = OCREngineKey.from_config(config)
        if self._engine is not None and self._engine_key == requested_key:
            return self._engine

        with self._engine_lock:
            if self._engine is None or self._engine_key != requested_key:
                self._engine = create_engine(config)
                self._engine_key = requested_key
            return self._engine

    def reset(self) -> None:
        """Clear cached state; intended for tests and controlled shutdown hooks."""

        with self._engine_lock:
            self._engine = None
            self._engine_key = None
