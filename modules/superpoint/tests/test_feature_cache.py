from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy

from PIL import Image

from superpoint.matching.feature_cache import FeatureCache
from superpoint.matching.superpoint_lightglue import SuperPointLightGlueMatcher


class Tensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, device):
        return self

    def clone(self):
        return Tensor(deepcopy(self.values))

    def numel(self):
        def count(value):
            return sum(count(item) for item in value) if isinstance(value, list) else 1
        return count(self.values)

    def element_size(self):
        return 4


class Torch:
    @staticmethod
    def inference_mode():
        return nullcontext()


def test_cache_copies_features_and_evicts_least_recently_used():
    cache = FeatureCache(8)
    original = {"x": Tensor([1])}
    cache.put("a", original, persistent=False)
    original["x"].values[0] = 99
    cache.put("b", {"x": Tensor([2])}, persistent=False)
    assert cache.get("a", Torch(), persistent=False)["x"].values == [1]
    cache.put("c", {"x": Tensor([3])}, persistent=False)
    assert cache.get("b", Torch(), persistent=False) is None
    assert cache.stats()["bytes"] == 8


def test_zero_budget_does_not_retain_query_features(tmp_path):
    cache = FeatureCache(0, tmp_path)
    cache.put("a", {"x": Tensor([1])}, persistent=False)
    assert cache.get("a", Torch(), persistent=False) is None
    assert not list(tmp_path.iterdir())


def test_query_extracted_once_and_matcher_cannot_mutate_cached_features():
    class Extractor:
        conf = {"test_weights": 1}

        def __init__(self):
            self.calls = []

        def extract(self, tensor, *, resize):
            self.calls.append(resize)
            return {"keypoints": Tensor([[[1, 2]]]), "descriptors": Tensor([[[3]]])}

    extractor = Extractor()

    def feature_match_fn(matcher, left, right):
        assert left["descriptors"].values == [[[3]]]
        left["descriptors"].values[0][0][0] = 999
        return ({"keypoints": left["keypoints"].values[0]},
                {"keypoints": right["keypoints"].values[0]},
                {"matches": [[0, 0]], "scores": [0.8]})

    matcher = SuperPointLightGlueMatcher(
        torch_module=Torch(), extractor=extractor, matcher=object(),
        image_to_tensor=lambda image: Tensor([1]), feature_match_fn=feature_match_fn,
        device="cpu", resize=128, feature_cache_mb=1, feature_cache_dir="",
    )
    query = Image.new("RGB", (8, 8), "red")
    matcher.match(query, Image.new("RGB", (8, 8), "blue"))
    matcher.match(query, Image.new("RGB", (8, 8), "green"))
    assert extractor.calls == [128, 128, 128]
    matcher.resize = 256
    matcher.match(query, Image.new("RGB", (8, 8), "blue"))
    assert extractor.calls == [128, 128, 128, 256, 256]
