"""DINOv2 image embedder with lazy optional ML imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from PIL import Image


class DinoV2Embedder:
    """Create normalized 384-dimensional embeddings with DINOv2-small."""

    MODEL_NAME = "facebook/dinov2-small"
    EMBEDDING_DIMENSION = 384

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        torch_module: Any | None = None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if (processor is None) != (model is None):
            raise ValueError("processor and model must be provided together")

        self._torch = (
            torch_module if torch_module is not None else import_module("torch")
        )
        self._model_name = model_name

        if processor is None and model is None:
            transformers = import_module("transformers")
            processor = transformers.AutoImageProcessor.from_pretrained(model_name)
            model = transformers.AutoModel.from_pretrained(model_name)

        self.processor = processor
        self.model = model
        self._device = self._torch.device(
            "cuda" if self._torch.cuda.is_available() else "cpu"
        )
        self.model.to(self._device)
        self.model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return str(self._device)

    def embed(self, image: Image.Image) -> list[float]:
        """Return an L2-normalized embedding for an RGB image."""

        processed_inputs = self.processor(images=image, return_tensors="pt")
        model_inputs = self._move_inputs_to_device(processed_inputs)

        with self._torch.inference_mode():
            outputs = self.model(**model_inputs)
            embedding = outputs.last_hidden_state[:, 0]
            embedding = self._torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1,
            )

        values = [float(value) for value in embedding.detach().cpu().flatten().tolist()]
        if len(values) != self.EMBEDDING_DIMENSION:
            raise ValueError(
                "DINOv2 embedding has unexpected dimension: "
                f"{len(values)}; expected {self.EMBEDDING_DIMENSION}"
            )
        return values

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self._device)

        return {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
