"""Domain exceptions raised by the OCR module."""


class OCRError(RuntimeError):
    """Base class for OCR module failures."""


class ImageDecodeError(OCRError):
    """Raised when an uploaded image cannot be decoded safely."""


class OCREngineError(OCRError):
    """Raised when an OCR backend fails."""


class ConfigurationError(OCRError):
    """Raised when OCR service configuration is invalid."""


class UnsupportedOCREngineError(OCRError):
    """Raised when the configured OCR backend is unknown."""
