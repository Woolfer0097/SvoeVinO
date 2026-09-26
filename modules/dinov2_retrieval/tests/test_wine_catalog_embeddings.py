from __future__ import annotations

from pathlib import Path

from dinov2_retrieval.wine_catalog.embeddings import wine_text
from dinov2_retrieval.wine_catalog.photo_manifest import (
    build_photo_manifest,
    photo_key,
    read_photo_manifest,
    stage_manifest_photos,
    write_photo_manifest,
)


def test_text_uses_all_csv_fields_in_stable_order() -> None:
    values = {
        "name": "Шардоне", "category": "Сухое", "color": "Белое",
        "region": "Крым", "grape_variety": "Шардоне",
        "description": "Фруктовый аромат", "winery": "Пример",
    }
    assert wine_text(values).splitlines() == [
        "Название: Шардоне", "Категория: Сухое", "Цвет: Белое",
        "Регион: Крым", "Сорт винограда: Шардоне",
        "Описание: Фруктовый аромат", "Винодельня: Пример",
    ]


def test_photo_manifest_matches_transliteration_and_exposes_ambiguity(tmp_path: Path) -> None:
    (tmp_path / "Spumante_belyj_bryut_d34e854a7b.webp").write_bytes(b"a")
    (tmp_path / "Arie_2020_KFB_742c588475.webp").write_bytes(b"b")
    (tmp_path / "Arie_2020_KFB_75504e4cd0.webp").write_bytes(b"c")
    (tmp_path / "thumbnail_Spumante_belyj_bryut_d34e854a7b.webp").write_bytes(b"d")

    assert photo_key("Спуманте белый брют.webp") == photo_key(
        "Spumante_belyj_bryut_d34e854a7b.webp"
    )
    rows = build_photo_manifest(
        ["Спуманте белый брют.webp", "Arie 2020 KFB.webp", "missing.webp"],
        tmp_path,
    )
    by_name = {row["dataset_photo"]: row for row in rows}
    assert by_name["Спуманте белый брют.webp"]["status"] == "matched"
    assert by_name["Arie 2020 KFB.webp"]["status"] == "ambiguous"
    assert by_name["Arie 2020 KFB.webp"]["candidate_count"] == "2"
    assert by_name["Arie 2020 KFB.webp"]["candidate_paths"].count("|") == 1
    assert by_name["missing.webp"]["status"] == "missing"

    output = tmp_path / "manifest.csv"
    write_photo_manifest(rows, output)
    assert read_photo_manifest(output) == {
        "Спуманте белый брют.webp": "Spumante_belyj_bryut_d34e854a7b.webp"
    }
    staged = tmp_path / "staged"
    summary = stage_manifest_photos(output, tmp_path, staged)
    assert summary == {"files": 3, "bytes": 3}
    assert (staged / "Spumante_belyj_bryut_d34e854a7b.webp").read_bytes() == b"a"
