"""Reference photo sources: a CSV manifest or a ``<slug>/*.jpg`` folder tree."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..config import get_data_root, get_supported_image_extensions, resolve_data_path
from ..contracts import ReferenceImage
from .csv_manifest import (
    ManifestError,
    container_path_problem,
    raise_for_problems,
    read_csv_manifest,
    resolve_data_root,
)

REQUIRED_COLUMNS = ("wine_id", "slug", "image_uri")
KIND = "Manifest"
DEFAULT_REFERENCE_DIR = "reference"
DEFAULT_MANIFEST_NAME = "manifest.csv"

__all__ = [
    "ManifestError",
    "REQUIRED_COLUMNS",
    "load_references",
    "read_reference_manifest",
    "scan_reference_directory",
]


def load_references(
    manifest: str | Path | None = None,
    reference_dir: str | Path | None = None,
) -> tuple[list[ReferenceImage], str]:
    """Return the reference photos and a description of where they came from.

    Relative paths are taken from DATA_ROOT. Without a manifest or a folder,
    DATA_ROOT/reference is used: its manifest.csv if there is one, otherwise
    the photos in the folder.
    """

    if manifest is not None and reference_dir is not None:
        raise ManifestError("Use either a manifest or a reference folder, not both")
    if manifest is not None:
        path = resolve_data_path(manifest)
        return read_reference_manifest(path), f"manifest {path}"
    if reference_dir is not None:
        folder = resolve_data_path(reference_dir)
        return scan_reference_directory(folder), f"folder {folder}"

    folder = get_data_root() / DEFAULT_REFERENCE_DIR
    default_manifest = folder / DEFAULT_MANIFEST_NAME
    if default_manifest.is_file():
        return read_reference_manifest(default_manifest), f"manifest {default_manifest}"
    return scan_reference_directory(folder), f"folder {folder}"


def read_reference_manifest(
    manifest_path: str | Path, data_root: str | Path | None = None
) -> list[ReferenceImage]:
    """Read and validate a CSV manifest with wine_id, slug and image_uri columns.

    Every row must fill all required columns; image_uri must be an absolute
    container path inside DATA_ROOT (``/data/...``) and unique; one wine_id
    must always use the same slug. All problems are reported together.
    """

    path = Path(manifest_path)
    root = resolve_data_root(data_root)
    rows, problems = read_csv_manifest(path, REQUIRED_COLUMNS, kind=KIND)

    references: list[ReferenceImage] = []
    line_by_image_uri: dict[str, int] = {}
    slug_by_wine_id: dict[str, tuple[str, int]] = {}
    for line, values in rows:
        image_uri = values["image_uri"]
        path_problem = container_path_problem(image_uri, root)
        if path_problem:
            problems.append(f"line {line}: image_uri {path_problem}")
            continue
        if image_uri in line_by_image_uri:
            problems.append(
                f"line {line}: duplicate image_uri {image_uri} "
                f"(first on line {line_by_image_uri[image_uri]})"
            )
            continue
        line_by_image_uri[image_uri] = line

        wine_id, slug = values["wine_id"], values["slug"]
        known_slug, known_line = slug_by_wine_id.setdefault(wine_id, (slug, line))
        if known_slug != slug:
            problems.append(
                f"line {line}: wine_id {wine_id} has slug {slug}, "
                f"but line {known_line} has slug {known_slug}"
            )
            continue

        references.append(ReferenceImage(**values))

    raise_for_problems(problems, path, KIND)
    if not references:
        raise ManifestError(f"Manifest has no reference photos: {path}")
    return references


def scan_reference_directory(
    reference_dir: str | Path,
    extensions: Iterable[str] | None = None,
) -> list[ReferenceImage]:
    """Collect reference photos from a folder; no manifest is needed.

    - ``<reference_dir>/<name>.<ext>`` — a photo in the folder itself is one
      wine: wine_id and slug are the file name without extension;
    - ``<reference_dir>/<wine>/<name>.<ext>`` — photos in a subfolder are one
      wine with several photos: wine_id and slug are the subfolder name.

    Hidden entries, deeper subfolders and unsupported extensions are skipped.
    """

    root = Path(reference_dir)
    if not root.is_dir():
        raise ManifestError(f"Reference directory does not exist: {root}")

    allowed_extensions = {
        extension.lower()
        for extension in (extensions or get_supported_image_extensions())
    }

    def is_photo(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in allowed_extensions

    references: list[ReferenceImage] = []
    for entry in _visible_entries(root):
        if entry.is_dir():
            references.extend(
                ReferenceImage(wine_id=entry.name, slug=entry.name, image_uri=str(photo))
                for photo in _visible_entries(entry)
                if is_photo(photo)
            )
        elif is_photo(entry):
            references.append(
                ReferenceImage(wine_id=entry.stem, slug=entry.stem, image_uri=str(entry))
            )

    if not references:
        raise ManifestError(
            f"No reference photos found in {root} or its subfolders "
            f"with extensions {', '.join(sorted(allowed_extensions))}"
        )
    return references


def _visible_entries(directory: Path) -> list[Path]:
    return sorted(
        entry for entry in directory.iterdir() if not entry.name.startswith(".")
    )
