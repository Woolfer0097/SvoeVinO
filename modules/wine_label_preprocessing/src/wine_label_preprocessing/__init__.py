"""CPU-first preprocessing for photographed wine labels."""

from .dewarpnet import DewarpNetAdapter
from .errors import (
    DewarpNetError,
    GeometryError,
    ImageInputError,
    ManifestError,
    PreprocessingError,
)
from .geometry import cylindrical_unwrap, estimate_cylinder_from_mask
from .image_io import crop_rgb, limit_resolution, load_rgb
from .models import (
    BBox,
    BenchmarkConfig,
    BenchmarkSample,
    ComparisonVariant,
    CylinderGeometry,
    DewarpNetConfig,
    DewarpNetResult,
    ImageAnnotations,
    OCRPrediction,
    OCRTruth,
    PhotometricConfig,
    PreprocessingConfig,
    PreprocessingResult,
    UnwrapConfig,
)
from .photometry import apply_photometric, mild_ocr_profile
from .pipeline import preprocess

__all__ = [
    "BBox",
    "BenchmarkConfig",
    "BenchmarkSample",
    "ComparisonVariant",
    "CylinderGeometry",
    "DewarpNetAdapter",
    "DewarpNetConfig",
    "DewarpNetError",
    "DewarpNetResult",
    "GeometryError",
    "ImageAnnotations",
    "ImageInputError",
    "ManifestError",
    "OCRPrediction",
    "OCRTruth",
    "PhotometricConfig",
    "PreprocessingConfig",
    "PreprocessingError",
    "PreprocessingResult",
    "UnwrapConfig",
    "apply_photometric",
    "crop_rgb",
    "cylindrical_unwrap",
    "estimate_cylinder_from_mask",
    "limit_resolution",
    "load_rgb",
    "mild_ocr_profile",
    "preprocess",
]

__version__ = "0.1.0"
