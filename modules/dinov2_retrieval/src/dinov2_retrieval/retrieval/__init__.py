"""Retrieval interfaces: vector search and reference embedding storage."""

from .base import Retriever
from .reference_repository import ReferenceRepository, RepositoryError, UpsertOutcome

__all__ = ["ReferenceRepository", "RepositoryError", "Retriever", "UpsertOutcome"]
