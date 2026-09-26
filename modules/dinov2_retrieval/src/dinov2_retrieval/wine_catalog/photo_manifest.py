"""Resolve CSV photo names to original Strapi uploads without guessing ambiguities."""

from __future__ import annotations

import csv
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

_TRANSLITERATION = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})
_VARIANT_PREFIX = re.compile(r"^(?:thumbnail|small|medium|large)_", re.I)
_STRAPI_HASH = re.compile(r"_[0-9a-f]{10}$", re.I)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def photo_key(filename: str) -> str:
    """Normalize a CSV/Strapi basename for conservative filename matching."""

    stem = Path(filename).stem
    stem = _VARIANT_PREFIX.sub("", stem)
    stem = _STRAPI_HASH.sub("", stem)
    stem = stem.casefold().translate(_TRANSLITERATION)
    return re.sub(r"[^a-z0-9]+", "", stem)


def build_photo_manifest(names: list[str], uploads_root: Path) -> list[dict[str, str]]:
    """Return one unambiguous path per photo name or an explicit failure status."""

    root = uploads_root.expanduser().resolve(strict=True)
    index: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in _IMAGE_EXTENSIONS:
            continue
        if _VARIANT_PREFIX.match(path.name):
            continue
        index[photo_key(path.name)].append(path)

    manifest = []
    for name in sorted(set(names)):
        candidates = index.get(photo_key(name), [])
        same_type = [path for path in candidates if path.suffix.casefold() == Path(name).suffix.casefold()]
        if same_type:
            candidates = same_type
        if len(candidates) == 1:
            status, relative = "matched", candidates[0].relative_to(root).as_posix()
        elif candidates:
            status, relative = "ambiguous", ""
        else:
            status, relative = "missing", ""
        manifest.append({
            "dataset_photo": name,
            "relative_path": relative,
            "status": status,
            "candidate_count": str(len(candidates)),
            "candidate_paths": "|".join(
                path.relative_to(root).as_posix() for path in candidates
            ) if status == "ambiguous" else "",
        })
    return manifest


def write_photo_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "dataset_photo", "relative_path", "status", "candidate_count",
                "candidate_paths",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def read_photo_manifest(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return {
            row["dataset_photo"]: row["relative_path"]
            for row in csv.DictReader(source)
            if row["status"] == "matched" and row["relative_path"]
        }


def stage_manifest_photos(manifest_path: Path, uploads_root: Path, output_root: Path) -> dict[str, int]:
    """Stage referenced originals and ambiguity candidates for portable export."""

    source_root = uploads_root.expanduser().resolve(strict=True)
    destination_root = output_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    paths: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["status"] == "matched" and row["relative_path"]:
                paths.add(row["relative_path"])
            elif row["status"] == "ambiguous" and row.get("candidate_paths"):
                paths.update(row["candidate_paths"].split("|"))
    total_bytes = 0
    for relative in sorted(paths):
        source_path = (source_root / relative).resolve(strict=True)
        target_path = (destination_root / relative).resolve()
        if not source_path.is_relative_to(source_root) or not target_path.is_relative_to(destination_root):
            raise ValueError(f"Photo path escapes the selected root: {relative}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)
        total_bytes += source_path.stat().st_size
    return {"files": len(paths), "bytes": total_bytes}
