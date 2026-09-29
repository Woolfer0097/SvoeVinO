"""DINOv2 image embedder with lazy optional ML imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from PIL import Image

from ..config import (
    DEFAULT_DINO_EMBEDDING_DIMENSION,
    DEFAULT_DINO_MODEL_NAME,
    get_dino_embedding_dimension,
    get_dino_model_name,
)
from .base import EmbeddingError


class DinoV2Embedder:
    """Create normalized embeddings with a DINOv2 model.

    The default is DINOv2-giant with registers (1536 values). The model name
    and the expected dimension default to ``DINO_MODEL_NAME`` and
    ``DINO_EMBEDDING_DIMENSION``.
    """

    MODEL_NAME = DEFAULT_DINO_MODEL_NAME
    EMBEDDING_DIMENSION = DEFAULT_DINO_EMBEDDING_DIMENSION

    def __init__(
        self,
        model_name: str | None = None,
        *,
        embedding_dimension: int | None = None,
        device: str | None = None,
        dtype: str = "float32",
        torch_module: Any | None = None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if (processor is None) != (model is None):
            raise ValueError("processor and model must be provided together")
        if dtype not in {"float32", "float16"}:
            raise ValueError("dtype must be float32 or float16")

        self._torch = (
            torch_module if torch_module is not None else import_module("torch")
        )
        self._model_name = get_dino_model_name(model_name)
        self._embedding_dimension = get_dino_embedding_dimension(embedding_dimension)
        self._dtype = self._torch.float16 if dtype == "float16" else None

        if processor is None and model is None:
            transformers = import_module("transformers")
            try:
                processor = transformers.AutoImageProcessor.from_pretrained(
                    self._model_name
                )
                model_options = (
                    {"torch_dtype": self._dtype, "low_cpu_mem_usage": True}
                    if self._dtype is not None else {}
                )
                model = transformers.AutoModel.from_pretrained(
                    self._model_name, **model_options
                )
            except (OSError, ValueError) as exc:
                raise EmbeddingError(
                    f"Cannot load model {self._model_name}: {exc}"
                ) from exc

        self.processor = processor
        self.model = model
        self._device = self._torch.device(
            device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        )
        if self._dtype is None:
            self.model.to(self._device)
        else:
            self.model.to(self._device, dtype=self._dtype)
        self.model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def embed(self, image: Image.Image) -> list[float]:
        """Return an L2-normalized embedding for an RGB image."""

        processed_inputs = self.processor(images=image, return_tensors="pt")
        model_inputs = self._move_inputs_to_device(processed_inputs)

        with self._torch.inference_mode():
            outputs = self.model(**model_inputs)
            # CLS token; with-registers models put their register tokens after it.
            embedding = outputs.last_hidden_state[:, 0]
            if self._dtype is not None:
                # Normalize in float32 even when the network uses half precision.
                embedding = embedding.float()
            embedding = self._torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1,
            )

        values = [float(value) for value in embedding.detach().cpu().flatten().tolist()]
        if len(values) != self._embedding_dimension:
            raise EmbeddingError(
                "DINOv2 embedding has unexpected dimension: "
                f"{len(values)}; expected {self._embedding_dimension}"
            )
        return values

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            if self._dtype is None:
                return inputs.to(self._device)
            return inputs.to(self._device, dtype=self._dtype)

        moved = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        if self._dtype is not None:
            moved = {
                key: value.to(dtype=self._dtype)
                if hasattr(value, "is_floating_point") and value.is_floating_point()
                else value
                for key, value in moved.items()
            }
        return moved
