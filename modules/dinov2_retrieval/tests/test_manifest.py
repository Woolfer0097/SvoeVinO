from __future__ import annotations

from pathlib import Path

import pytest

from dinov2_retrieval.contracts import ReferenceImage
from dinov2_retrieval.infrastructure.reference_manifest import (
    ManifestError,
    read_reference_manifest,
    scan_reference_directory,
)

DATA_ROOT = "/data"

VALID_MANIFEST = """wine_id,slug,image_uri
wine-001,cabernet-2020,/data/reference/wine-001/photo-1.jpg
wine-001,cabernet-2020,/data/reference/wine-001/photo-2.jpg
wine-002,merlot-2021,/data/reference/wine-002/photo-1.jpg
"""


def write_manifest(tmp_path: Path, content: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text(content, encoding=encoding)
    return path


def read(path: Path):
    return read_reference_manifest(path, data_root=DATA_ROOT)


def test_reads_reference_photos_in_order(tmp_path: Path) -> None:
    references = read(write_manifest(tmp_path, VALID_MANIFEST))

    assert references == [
        ReferenceImage(
            wine_id="wine-001",
            slug="cabernet-2020",
            image_uri="/data/reference/wine-001/photo-1.jpg",
        ),
        ReferenceImage(
            wine_id="wine-001",
            slug="cabernet-2020",
            image_uri="/data/reference/wine-001/photo-2.jpg",
        ),
        ReferenceImage(
            wine_id="wine-002",
            slug="merlot-2021",
            image_uri="/data/reference/wine-002/photo-1.jpg",
        ),
    ]


def test_tolerates_bom_spaces_extra_columns_and_blank_rows(tmp_path: Path) -> None:
    content = (
        "image_uri , note, wine_id ,slug\n"
        " /data/reference/a.jpg , front , wine-001 , cabernet-2020 \n"
        "\n"
        ",,,\n"
    )
    path = write_manifest(tmp_path, content, encoding="utf-8-sig")

    assert read(path) == [
        ReferenceImage(
            wine_id="wine-001", slug="cabernet-2020", image_uri="/data/reference/a.jpg"
        )
    ]


def test_missing_required_columns(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, "wine_id;slug;image_uri\nw;s;/data/a.jpg\n")

    with pytest.raises(ManifestError, match="missing required columns: wine_id, slug, image_uri"):
        read(path)


@pytest.mark.parametrize(
    ("row", "empty_column"),
    [
        (",cabernet-2020,/data/a.jpg", "wine_id"),
        ("wine-001, ,/data/a.jpg", "slug"),
        ("wine-001,cabernet-2020,", "image_uri"),
        ("wine-001,cabernet-2020", "image_uri"),
    ],
)
def test_required_values_must_be_filled(
    tmp_path: Path, row: str, empty_column: str
) -> None:
    path = write_manifest(tmp_path, f"wine_id,slug,image_uri\n{row}\n")

    with pytest.raises(ManifestError, match=f"line 2: empty {empty_column}"):
        read(path)


def test_duplicate_image_uri_is_rejected(tmp_path: Path) -> None:
    content = VALID_MANIFEST + "wine-003,syrah-2019,/data/reference/wine-001/photo-2.jpg\n"

    with pytest.raises(
        ManifestError,
        match=r"line 5: duplicate image_uri /data/reference/wine-001/photo-2.jpg \(first on line 3\)",
    ):
        read(write_manifest(tmp_path, content))


def test_one_wine_must_keep_one_slug(tmp_path: Path) -> None:
    content = VALID_MANIFEST + "wine-002,merlot-2022,/data/reference/wine-002/photo-2.jpg\n"

    with pytest.raises(ManifestError, match="wine_id wine-002 has slug merlot-2022"):
        read(write_manifest(tmp_path, content))


def test_all_problems_are_reported_together(tmp_path: Path) -> None:
    content = (
        "wine_id,slug,image_uri\n,s,/data/a.jpg\nw,,/data/b.jpg\nw,s,/data/a.jpg,extra\n"
    )

    with pytest.raises(ManifestError) as error:
        read(write_manifest(tmp_path, content))

    message = str(error.value)
    assert "line 2: empty wine_id" in message
    assert "line 3: empty slug" in message
    assert "line 4: 4 values for 3 columns" in message


@pytest.mark.parametrize(
    ("content", "message"),
    [("", "Manifest is empty"), ("wine_id,slug,image_uri\n", "has no reference photos")],
)
def test_manifest_without_photos(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        read(write_manifest(tmp_path, content))


def test_missing_manifest_file(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="Manifest does not exist"):
        read(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    ("image_uri", "message"),
    [
        ("reference/wine-001/photo-1.jpg", "must be absolute and start with /data"),
        ("/tmp/photo-1.jpg", "must be inside DATA_ROOT /data"),
        ("/data/../etc/photo.jpg", "must be inside DATA_ROOT /data"),
        ("/database/photo.jpg", "must be inside DATA_ROOT /data"),
    ],
)
def test_image_uri_must_be_a_container_path_inside_data_root(
    tmp_path: Path, image_uri: str, message: str
) -> None:
    content = f"wine_id,slug,image_uri\nwine-001,cabernet-2020,{image_uri}\n"

    with pytest.raises(ManifestError, match=f"line 2: image_uri path {message}"):
        read(write_manifest(tmp_path, content))


def test_data_root_is_taken_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    content = f"wine_id,slug,image_uri\nwine-001,cabernet-2020,{tmp_path}/a.jpg\n"

    references = read_reference_manifest(write_manifest(tmp_path, content))

    assert references[0].image_uri == f"{tmp_path}/a.jpg"


def test_image_uri_through_a_symlinked_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_root = tmp_path / "real-data"
    real_root.mkdir()
    link_root = tmp_path / "data"
    link_root.symlink_to(real_root)
    monkeypatch.setenv("DATA_ROOT", str(link_root))
    content = f"wine_id,slug,image_uri\nwine-001,cabernet-2020,{link_root}/a.jpg\n"

    references = read_reference_manifest(write_manifest(tmp_path, content))

    assert references[0].image_uri == f"{link_root}/a.jpg"


def write_files(root: Path, *relative_paths: str) -> None:
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")


def scanned(root: Path) -> list[tuple[str, str, str]]:
    return [
        (ref.wine_id, ref.slug, str(Path(ref.image_uri).relative_to(root)))
        for ref in scan_reference_directory(root)
    ]


def test_scan_uses_folder_name_as_wine_id_and_slug(tmp_path: Path) -> None:
    write_files(
        tmp_path,
        "merlot-2021/photo-1.JPG",
        "cabernet-2020/photo-2.png",
        "cabernet-2020/photo-1.jpg",
        "cabernet-2020/notes.txt",
        "cabernet-2020/.hidden.jpg",
        "cabernet-2020/deeper/photo-3.jpg",
        ".cache/photo.jpg",
    )

    assert scanned(tmp_path) == [
        ("cabernet-2020", "cabernet-2020", "cabernet-2020/photo-1.jpg"),
        ("cabernet-2020", "cabernet-2020", "cabernet-2020/photo-2.png"),
        ("merlot-2021", "merlot-2021", "merlot-2021/photo-1.JPG"),
    ]


def test_scan_photos_in_the_folder_itself_are_one_wine_each(tmp_path: Path) -> None:
    write_files(
        tmp_path,
        "87.6_20-08-2026_19-56-56.webp",
        "1.73_06-09-2026_14-54-06.WEBP",
        "manifest.csv",
        "notes.txt",
        ".hidden.webp",
    )

    assert scanned(tmp_path) == [
        ("1.73_06-09-2026_14-54-06", "1.73_06-09-2026_14-54-06", "1.73_06-09-2026_14-54-06.WEBP"),
        ("87.6_20-08-2026_19-56-56", "87.6_20-08-2026_19-56-56", "87.6_20-08-2026_19-56-56.webp"),
    ]


def test_scan_mixes_single_photos_and_wine_folders(tmp_path: Path) -> None:
    write_files(tmp_path, "donum-2023/front.webp", "donum-2023/side.jpg", "single.png")

    assert scanned(tmp_path) == [
        ("donum-2023", "donum-2023", "donum-2023/front.webp"),
        ("donum-2023", "donum-2023", "donum-2023/side.jpg"),
        ("single", "single", "single.png"),
    ]


def test_scan_without_photos_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty-wine").mkdir()

    with pytest.raises(ManifestError, match="No reference photos found"):
        scan_reference_directory(tmp_path)


def test_scan_of_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="Reference directory does not exist"):
        scan_reference_directory(tmp_path / "missing")
