import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wine_label_preprocessing.api import create_app, service_config
from wine_label_preprocessing.models import PreprocessingConfig
from wine_label_preprocessing.photometry import mild_ocr_profile


def png(size=(300, 400), color=(45, 55, 65)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def test_http_adapter_reuses_safe_independent_branches():
    config = PreprocessingConfig(embedding_max_long_side=200, ocr_max_long_side=300,
                                 ocr_photometric=mild_ocr_profile(unsharp=True))
    with TestClient(create_app(config)) as client:
        response = client.post("/preprocess", files={"image": ("photo.png", png())})
    assert response.status_code == 200
    result = response.json()
    visual = Image.open(io.BytesIO(base64.b64decode(result["embedding_png"])))
    ocr = Image.open(io.BytesIO(base64.b64decode(result["ocr_png"])))
    assert visual.size == (150, 200)
    assert ocr.size == (225, 300)
    assert result["metadata"]["embedding"]["photometric_operations"] == []
    assert {op["name"] for op in result["metadata"]["ocr"]["photometric_operations"]} == {
        "brightness", "clahe", "unsharp",
    }
    assert result["metadata"]["cylindrical"]["reason"] == "label_bbox_missing"


def test_small_images_are_not_upscaled_or_changed_in_visual_branch():
    with TestClient(create_app()) as client:
        result = client.post("/preprocess", files={"image": ("small.png", png((30, 40)))}).json()
    visual = Image.open(io.BytesIO(base64.b64decode(result["embedding_png"])))
    assert visual.size == (30, 40)
    assert visual.getpixel((10, 10)) == (45, 55, 65)


@pytest.mark.parametrize("data", [b"", b"broken"])
def test_invalid_upload_is_rejected(data):
    with TestClient(create_app()) as client:
        assert client.post("/preprocess", files={"image": ("bad.jpg", data)}).status_code == 422


def test_service_config_bounds_and_opt_out(monkeypatch):
    monkeypatch.setenv("PREPROCESS_VISUAL_MAX_SIDE", "4096")
    with pytest.raises(ValueError):
        service_config()
    monkeypatch.setenv("PREPROCESS_VISUAL_MAX_SIDE", "1600")
    monkeypatch.setenv("PREPROCESS_OCR_MILD", "false")
    assert not service_config().ocr_photometric.clahe_enabled
