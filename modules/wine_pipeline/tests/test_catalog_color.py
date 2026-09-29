import io

import pytest
from PIL import Image

from wine_pipeline.catalog import apply_vintage, make_card
from wine_pipeline.color import blend, palette, similarity
from wine_pipeline.fusion import fuse


def test_card_keeps_all_text_without_vectors_or_import_details():
    row = ("label-2023", " Wine 2023 ", "Red", "Ruby", "Crimea", "Merlot", " A long description ",
           "Winery", "wine.webp", {"Описание": "duplicate", "Alcohol": "13%", "Название фото": "secret.jpg"})
    card = make_card(row)
    assert card["description"] == "A long description"
    assert card["vintage"] == 2023
    assert card["web_photo_uri"] == "/data/web_photos/wine.webp"
    assert card["attributes"] == [{"label": "Alcohol", "value": "13%"}]


@pytest.mark.parametrize("path", ["../secret.jpg", "path/photo.jpg", r"path\photo.jpg"])
def test_photo_paths_cannot_escape_web_root(path):
    assert make_card(("wine", *[""] * 7, path, {}))["web_photo_uri"] is None


def test_description_foundation_year_is_not_a_catalog_vintage():
    assert make_card(("wine", "Wine", *[""] * 4, "Founded in 1950", "", None, {}))["vintage"] is None


def test_exact_year_promotes_matching_candidate_over_wrong_vintage():
    ranked = [{"slug": "wine-2021", "vintage": 2021, "score": .99},
              {"slug": "wine-2023", "vintage": 2023, "score": .98}]
    result, check = apply_vintage(ranked, [2023])
    assert result[0]["slug"] == "wine-2023"
    assert check == {"detected_year": 2023, "state": "matched"}


def test_unknown_year_is_not_invented_and_multiple_years_are_ambiguous():
    ranked = [{"slug": "wine", "score": .8}]
    assert apply_vintage(ranked, [2023])[1]["state"] == "unverified"
    assert apply_vintage(ranked, [2022, 2023])[1]["state"] == "ambiguous"


def image(color):
    buffer = io.BytesIO()
    Image.new("RGBA", (80, 120), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_color_uses_image_palette_not_wine_type():
    red = palette(image((220, 30, 50, 255)))
    pink = palette(image((235, 110, 130, 255)))
    blue = palette(image((25, 50, 210, 255)))
    assert similarity(red, pink) > similarity(red, blue)
    assert palette(image((0, 0, 0, 0))) is None


def test_five_percent_soft_color_reranking_does_not_delete_candidates():
    ranked = [{"slug": "different-color", "score": .9}, {"slug": "same-color", "score": .89}]
    result = blend(ranked, {"different-color": 0, "same-color": 1}, .05)
    assert result[0]["slug"] == "same-color"
    assert len(result) == 2
    assert result[0]["score"] == pytest.approx(.95 * .89 + .05)


def test_requested_ocr_weight_is_configurable_and_validated():
    candidates = [{"slug": "a", "score": .9}, {"slug": "b", "score": .8}]
    result, _ = fuse(candidates, {"a": 1, "b": .8}, {"a": .7, "b": 1}, ocr_weight=.5)
    assert result[0]["slug"] == "b"
    with pytest.raises(ValueError):
        fuse(candidates, {}, {}, ocr_weight=1.5)
