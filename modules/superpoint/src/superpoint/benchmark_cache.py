"""Local CPU timing comparison; no organizer data or running API required."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageOps

from .matching.feature_cache import FeatureCache
from .matching.superpoint_lightglue import SuperPointLightGlueMatcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    with Image.open(args.query) as source:
        query = ImageOps.exif_transpose(source).convert("RGB")
    references = []
    for path in args.references:
        with Image.open(path) as source:
            references.append(ImageOps.exif_transpose(source).convert("RGB"))
    with tempfile.TemporaryDirectory(prefix="superpoint-benchmark-") as directory:
        cached = SuperPointLightGlueMatcher(device="cpu", feature_cache_dir=directory)
        legacy = SuperPointLightGlueMatcher(
            torch_module=cached._torch, extractor=cached.extractor, matcher=cached.matcher,
            image_to_tensor=cached._image_to_tensor, match_fn=cached._match_fn,
            device="cpu", max_num_keypoints=cached.max_num_keypoints, resize=cached.resize,
        )
        legacy.match(Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64)))
        calls = 0
        original_extract = cached.extractor.extract

        def extract(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_extract(*args, **kwargs)

        cached.extractor.extract = extract
        baseline = None
        phases = (("uncached_first", legacy), ("cached_cold", cached),
                  ("cached_warm_references", cached), ("uncached_after", legacy))
        for label, matcher in phases:
            if label == "cached_warm_references":
                # Simulate a new upload after restart: only reference features
                # survive on disk. The query must be extracted once again.
                cached.feature_cache = FeatureCache(
                    cached.feature_cache.max_bytes, Path(directory))
            count_before = calls
            started = time.perf_counter()
            predictions = [matcher.match(query, reference) for reference in references]
            seconds = time.perf_counter() - started
            signatures = [(p.query_points, p.reference_points, p.scores) for p in predictions]
            if baseline is None:
                baseline = signatures
            print(json.dumps({"phase": label, "seconds": round(seconds, 6),
                              "device": matcher.device, "threads": cached._torch.get_num_threads(),
                              "resize": cached.resize, "max_keypoints": cached.max_num_keypoints,
                              "pairs": len(predictions), "extractions": calls - count_before,
                              "same_correspondences": signatures == baseline,
                              "cache": cached.feature_cache.stats()}, ensure_ascii=False), flush=True)
    query.close()
    for reference in references:
        reference.close()


if __name__ == "__main__":
    main()
