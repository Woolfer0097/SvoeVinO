"""Package-specific errors for invalid preprocessing inputs and configuration."""

from __future__ import annotations


class PreprocessingError(ValueError):
    """Base error raised for invalid input or configuration."""


class ImageInputError(PreprocessingError):
    """Raised when an image, bounding box, or mask cannot be interpreted safely."""


class GeometryError(PreprocessingError):
    """Raised by strict geometry APIs for invalid cylindrical parameters."""


class ManifestError(PreprocessingError):
    """Raised when a comparison manifest is inconsistent."""


class DewarpNetError(PreprocessingError):
    """Raised when strict DewarpNet loading or inference fails."""
