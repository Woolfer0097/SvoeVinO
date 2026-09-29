"""Bounded CPU feature cache; only reference features may persist on disk."""
from __future__ import annotations

import logging
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

logger = logging.getLogger(__name__)


class FeatureCache:
    def __init__(self, max_bytes: int, directory: Path | None = None) -> None:
        if max_bytes < 0:
            raise ValueError("Feature cache size must not be negative")
        self.max_bytes = max_bytes
        self.directory = directory
        self.entries: OrderedDict[str, tuple[dict[str, Any], int]] = OrderedDict()
        self.bytes_used = 0
        self.hits = self.disk_hits = self.misses = 0

    def get(self, key: str, torch, *, persistent: bool) -> dict | None:
        if not self.max_bytes:
            self.misses += 1
            return None
        item = self.entries.get(key)
        if item is not None:
            self.entries.move_to_end(key)
            self.hits += 1
            return item[0]
        if persistent and self.directory is not None:
            try:
                import numpy as np

                with np.load(self.directory / (key + ".npz"), allow_pickle=False) as data:
                    required = {"keypoints", "keypoint_scores", "descriptors", "image_size"}
                    if set(data.files) != required:
                        raise ValueError("Unexpected cached feature fields")
                    arrays = {name: data[name] for name in required}
                    points = arrays["keypoints"]
                    count = points.shape[1] if points.ndim == 3 else -1
                    if (points.shape != (1, count, 2) or not 0 <= count <= 100_000
                            or arrays["descriptors"].shape != (1, count, 256)
                            or arrays["keypoint_scores"].shape != (1, count)
                            or arrays["image_size"].shape != (1, 2)
                            or any(a.dtype != np.float32 or not np.isfinite(a).all()
                                   for a in arrays.values())):
                        raise ValueError("Invalid cached feature shapes or values")
                    features = {name: torch.from_numpy(array.copy())
                                for name, array in arrays.items()}
                self._remember(key, features)
                self.disk_hits += 1
                return features
            except FileNotFoundError:
                pass
            except (OSError, ValueError, EOFError, BadZipFile):
                logger.warning("Ignoring unreadable SuperPoint feature cache entry %s", key)
        self.misses += 1
        return None

    def put(self, key: str, features: dict, *, persistent: bool) -> dict:
        # Do not retain GPU memory; copying is blocking before arrays are read.
        cpu = {name: value.detach().cpu().clone() for name, value in features.items()}
        if not self.max_bytes:
            return cpu
        self._remember(key, cpu)
        if persistent and self.directory is not None:
            temporary = None
            try:
                import numpy as np

                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".npz", delete=False) as output:
                    temporary = Path(output.name)
                    np.savez_compressed(output, **{name: value.numpy() for name, value in cpu.items()})
                os.replace(temporary, self.directory / (key + ".npz"))
            except OSError:
                logger.warning("Cannot persist SuperPoint features; using RAM cache", exc_info=True)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return cpu

    def _remember(self, key: str, features: dict) -> None:
        size = sum(value.numel() * value.element_size() for value in features.values())
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes_used -= previous[1]
        if size > self.max_bytes:
            return
        while self.entries and self.bytes_used + size > self.max_bytes:
            _, (_, removed) = self.entries.popitem(last=False)
            self.bytes_used -= removed
        self.entries[key] = (features, size)
        self.bytes_used += size

    def stats(self) -> dict[str, int]:
        return {"entries": len(self.entries), "bytes": self.bytes_used,
                "max_bytes": self.max_bytes, "hits": self.hits,
                "disk_hits": self.disk_hits, "misses": self.misses}
