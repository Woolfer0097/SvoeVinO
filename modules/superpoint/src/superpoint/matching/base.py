"""Matcher boundary used by photo verification."""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from ..contracts import MatchPrediction


class PhotoMatcher(Protocol):
    """Match local features between a query photo and a reference photo."""

    @property
    def model_name(self) -> str:
        """Return the matcher identifier."""

        ...

    @property
    def device(self) -> str:
        """Return the runtime device identifier."""

        ...

    def match(self, query_image: Image.Image, reference_image: Image.Image) -> MatchPrediction:
        """Return correspondences between two RGB images."""

        ...
