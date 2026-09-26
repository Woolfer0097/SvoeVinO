"""Application use cases."""

from .check_health import check_health
from .create_embedding import create_embedding
from .evaluate_retrieval import compute_recall, evaluate_retrieval
from .index_reference_images import index_reference_images
from .search_similar_wines import search_similar_wines, select_top_wines
from .validate_image import validate_image

__all__ = [
    "check_health",
    "compute_recall",
    "create_embedding",
    "evaluate_retrieval",
    "index_reference_images",
    "search_similar_wines",
    "select_top_wines",
    "validate_image",
]
