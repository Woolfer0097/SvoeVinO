from __future__ import annotations

import sys
from contextlib import nullcontext

import pytest
from PIL import Image

from superpoint.matching.superpoint_lightglue import (
    SuperPointLightGlueMatcher,
    prediction_from_features,
)


class ListTensor:
    def __init__(self, values: list) -> None:
        self.values = values

    def detach(self) -> ListTensor:
        return self

    def cpu(self) -> ListTensor:
        return self

    def tolist(self) -> list:
        return self.values


class MovedImage:
    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.device: str | None = None

    def to(self, device: str) -> MovedImage:
        self.device = device
        return self


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    @staticmethod
    def inference_mode():
        return nullcontext()


def test_prediction_keeps_only_matched_points() -> None:
    prediction = prediction_from_features(
        {"keypoints": ListTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        {"keypoints": ListTensor([[10.0, 11.0], [12.0, 13.0]])},
        {
            "matches": ListTensor([[0, 1], [2, 0]]),
            "scores": ListTensor([0.8, 0.4]),
        },
    )

    assert prediction.query_points == [(1.0, 2.0), (5.0, 6.0)]
    assert prediction.reference_points == [(12.0, 13.0), (10.0, 11.0)]
    assert prediction.scores == [0.8, 0.4]
    assert prediction.num_keypoints_query == 3
    assert prediction.num_keypoints_reference == 2


def test_prediction_rejects_an_index_outside_the_keypoints() -> None:
    with pytest.raises(RuntimeError, match="out-of-range"):
        prediction_from_features(
            {"keypoints": [[0.0, 0.0]]},
            {"keypoints": [[1.0, 1.0]]},
            {"matches": [[0, 3]], "scores": [0.5]},
        )


def test_matcher_passes_resize_and_original_images_to_lightglue() -> None:
    seen: dict[str, object] = {}

    def image_to_tensor(image: Image.Image) -> MovedImage:
        return MovedImage(image)

    def match_fn(extractor, matcher, image0, image1, device, resize):
        seen["extractor"] = extractor
        seen["matcher"] = matcher
        seen["devices"] = (image0.device, image1.device)
        seen["result_device"] = device
        seen["resize"] = resize
        return (
            {"keypoints": [[0.0, 1.0]]},
            {"keypoints": [[2.0, 3.0]]},
            {"matches": [[0, 0]], "scores": [0.6]},
        )

    photo_matcher = SuperPointLightGlueMatcher(
        torch_module=FakeTorch(),
        extractor="extractor",
        matcher="matcher",
        image_to_tensor=image_to_tensor,
        match_fn=match_fn,
        device="cpu",
        max_num_keypoints=32,
        resize=128,
        depth_confidence=0.5,
        width_confidence=0.5,
        filter_threshold=0.2,
    )

    prediction = photo_matcher.match(
        Image.new("RGB", (4, 4)),
        Image.new("RGB", (4, 4)),
    )

    assert photo_matcher.model_name == "superpoint+lightglue"
    assert photo_matcher.device == "cpu"
    assert seen == {
        "extractor": "extractor",
        "matcher": "matcher",
        "devices": ("cpu", "cpu"),
        "result_device": "cpu",
        "resize": 128,
    }
    assert prediction.num_matches == 1
    assert prediction.query_points == [(0.0, 1.0)]


def test_cuda_request_fails_when_cuda_is_unavailable() -> None:
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        SuperPointLightGlueMatcher(
            torch_module=FakeTorch(),
            extractor=object(),
            matcher=object(),
            image_to_tensor=lambda image: image,
            match_fn=lambda *args, **kwargs: ({}, {}, {}),
            device="cuda",
        )


def test_cuda_results_are_synchronized_before_reading_cpu_lists():
    pending = {"copied": False}

    class CudaTorch(FakeTorch):
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def synchronize(device):
                assert device == "cuda"
                pending["copied"] = True

    class AsyncTensor(ListTensor):
        def tolist(self):
            assert pending["copied"], "GPU to CPU copy has not finished"
            return super().tolist()

    matcher = SuperPointLightGlueMatcher(
        torch_module=CudaTorch(), extractor=object(), matcher=object(),
        image_to_tensor=MovedImage, device="cuda",
        match_fn=lambda *args, **kwargs: (
            {"keypoints": AsyncTensor([[1, 2]])},
            {"keypoints": AsyncTensor([[3, 4]])},
            {"matches": AsyncTensor([[0, 0]]), "scores": AsyncTensor([.8])},
        ),
    )
    assert matcher.match(Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))).num_matches == 1


def test_negative_match_indices_are_not_python_list_offsets():
    with pytest.raises(RuntimeError, match="out-of-range"):
        prediction_from_features(
            {"keypoints": [[0, 0]]}, {"keypoints": [[0, 0]]},
            {"matches": [[-1, 0]], "scores": [.5]},
        )


def test_missing_ml_dependencies_have_an_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("lightglue"):
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "lightglue", raising=False)
    monkeypatch.setattr("builtins.__import__", blocked_import)

    with pytest.raises(ImportError, match=r"pip install -e '.\[ml\]'"):
        SuperPointLightGlueMatcher()
