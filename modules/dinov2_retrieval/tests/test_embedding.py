from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from PIL import Image

from dinov2_retrieval.embedding.base import EmbeddingError
from dinov2_retrieval.embedding.dinov2_embedder import DinoV2Embedder

DIMENSION = 1536


class FakeTensor:
    def __init__(self, values, device: str = "cpu") -> None:
        self.values = values
        self.device = device

    def __getitem__(self, item):
        if isinstance(item, tuple):
            batch_selector, token_index = item
            rows = self.values[batch_selector]
            return FakeTensor([row[token_index] for row in rows], self.device)
        return self.values[item]

    def to(self, device):
        self.device = str(device)
        return self

    def detach(self):
        return self

    def cpu(self):
        self.device = "cpu"
        return self

    def flatten(self):
        if self.values and isinstance(self.values[0], list):
            return FakeTensor(
                [value for row in self.values for value in row], self.device
            )
        return self

    def tolist(self):
        return self.values


class FakeFunctional:
    @staticmethod
    def normalize(tensor: FakeTensor, p: int, dim: int) -> FakeTensor:
        assert p == 2
        assert dim == 1
        rows = tensor.values
        if rows and not isinstance(rows[0], list):
            rows = [rows]
        normalized = []
        for row in rows:
            norm = math.sqrt(sum(value * value for value in row))
            normalized.append([value / norm for value in row])
        return FakeTensor(normalized, tensor.device)


class FakeInferenceMode:
    def __init__(self, torch_module) -> None:
        self.torch_module = torch_module

    def __enter__(self):
        self.torch_module.inference_mode_calls += 1

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeTorch:
    def __init__(self, cuda_available: bool) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self.nn = SimpleNamespace(functional=FakeFunctional())
        self.inference_mode_calls = 0

    def device(self, name: str) -> str:
        return name

    def inference_mode(self):
        return FakeInferenceMode(self)


class FakeProcessor:
    def __init__(self) -> None:
        self.calls = 0
        self.received_image = None

    def __call__(self, *, images, return_tensors):
        self.calls += 1
        self.received_image = images
        assert return_tensors == "pt"
        return {"pixel_values": FakeTensor([[0.0]])}


class FakeModel:
    def __init__(self) -> None:
        self.to_device = None
        self.eval_calls = 0
        self.calls = 0
        self.received_inputs = None

    def to(self, device):
        self.to_device = str(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, **inputs):
        self.calls += 1
        self.received_inputs = inputs
        # CLS first, then register tokens, as in DINOv2 with registers.
        cls_token = [3.0, 4.0] + [0.0] * (DIMENSION - 2)
        register_token = [0.0, 0.0, 1.0] + [0.0] * (DIMENSION - 3)
        tokens = [cls_token, register_token, register_token]
        return SimpleNamespace(last_hidden_state=FakeTensor([tokens]))


@pytest.mark.parametrize(
    ("cuda_available", "expected_device"),
    [(True, "cuda"), (False, "cpu")],
)
def test_dinov2_embedding_dimension_normalization_and_device(
    cuda_available: bool, expected_device: str
) -> None:
    torch_module = FakeTorch(cuda_available)
    processor = FakeProcessor()
    model = FakeModel()
    embedder = DinoV2Embedder(
        torch_module=torch_module,
        processor=processor,
        model=model,
    )

    image = Image.new("RGB", (8, 4), color=(20, 40, 60))
    embedding = embedder.embed(image)

    assert len(embedding) == DIMENSION
    assert embedding[:3] == pytest.approx([0.6, 0.8, 0.0])
    assert math.sqrt(sum(value * value for value in embedding)) == pytest.approx(1.0)
    assert embedder.device == expected_device
    assert model.to_device == expected_device
    assert model.eval_calls == 1
    assert model.calls == 1
    assert processor.calls == 1
    assert processor.received_image is image
    assert model.received_inputs["pixel_values"].device == expected_device
    assert torch_module.inference_mode_calls == 1


def test_dinov2_model_and_processor_load_once(monkeypatch) -> None:
    from dinov2_retrieval.embedding import dinov2_embedder as module

    torch_module = FakeTorch(cuda_available=False)
    processor = FakeProcessor()
    model = FakeModel()
    processor_loads = 0
    model_loads = 0

    class AutoImageProcessor:
        @classmethod
        def from_pretrained(cls, model_name):
            nonlocal processor_loads
            assert model_name == "facebook/dinov2-with-registers-giant"
            processor_loads += 1
            return processor

    class AutoModel:
        @classmethod
        def from_pretrained(cls, model_name):
            nonlocal model_loads
            assert model_name == "facebook/dinov2-with-registers-giant"
            model_loads += 1
            return model

    fake_transformers = SimpleNamespace(
        AutoImageProcessor=AutoImageProcessor,
        AutoModel=AutoModel,
    )

    def fake_import(name: str):
        assert name == "transformers"
        return fake_transformers

    monkeypatch.setattr(module, "import_module", fake_import)
    embedder = DinoV2Embedder(torch_module=torch_module)

    image = Image.new("RGB", (8, 4))
    embedder.embed(image)
    embedder.embed(image)

    assert processor_loads == 1
    assert model_loads == 1
    assert model.calls == 2


def fake_transformers(processor, model, loaded_names: list[str]):
    class AutoImageProcessor:
        @classmethod
        def from_pretrained(cls, model_name):
            loaded_names.append(model_name)
            return processor

    class AutoModel:
        @classmethod
        def from_pretrained(cls, model_name):
            loaded_names.append(model_name)
            return model

    return SimpleNamespace(AutoImageProcessor=AutoImageProcessor, AutoModel=AutoModel)


def test_model_name_and_dimension_come_from_config(monkeypatch) -> None:
    from dinov2_retrieval.embedding import dinov2_embedder as module

    monkeypatch.setenv("DINO_MODEL_NAME", "local/dinov2-custom")
    monkeypatch.setenv("DINO_EMBEDDING_DIMENSION", "384")
    loaded_names: list[str] = []
    transformers = fake_transformers(FakeProcessor(), FakeModel(), loaded_names)
    monkeypatch.setattr(module, "import_module", lambda name: transformers)

    embedder = DinoV2Embedder(torch_module=FakeTorch(cuda_available=False))

    assert loaded_names == ["local/dinov2-custom", "local/dinov2-custom"]
    assert embedder.model_name == "local/dinov2-custom"
    assert embedder.embedding_dimension == 384


def test_unexpected_dimension_raises_embedding_error(monkeypatch) -> None:
    monkeypatch.setenv("DINO_EMBEDDING_DIMENSION", "768")
    embedder = DinoV2Embedder(
        torch_module=FakeTorch(cuda_available=False),
        processor=FakeProcessor(),
        model=FakeModel(),
    )

    with pytest.raises(EmbeddingError, match="unexpected dimension: 1536; expected 768"):
        embedder.embed(Image.new("RGB", (8, 4)))


def test_model_load_failure_raises_embedding_error(monkeypatch) -> None:
    from dinov2_retrieval.embedding import dinov2_embedder as module

    class AutoImageProcessor:
        @classmethod
        def from_pretrained(cls, model_name):
            raise OSError("not in the cache and no internet connection")

    monkeypatch.setattr(
        module,
        "import_module",
        lambda name: SimpleNamespace(AutoImageProcessor=AutoImageProcessor),
    )

    with pytest.raises(EmbeddingError, match="Cannot load model facebook/dinov2-with-registers-giant"):
        DinoV2Embedder(torch_module=FakeTorch(cuda_available=False))
