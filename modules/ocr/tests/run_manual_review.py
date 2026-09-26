"""Run the REST OCR service on a folder of photos for manual review.

Only the output directory is modified. Source photos are copied with stable
imageN names so every JSON/TXT pair can be traced back to its source.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def next_run_dir(outputs: Path) -> Path:
    outputs.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(path.name[4:])
        for path in outputs.iterdir()
        if path.is_dir() and path.name.startswith("test") and path.name[4:].isdigit()
    ]
    run_dir = outputs / f"test{max(numbers, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


def save_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_run(source_dir: Path, outputs: Path) -> tuple[Path, dict]:
    photos = sorted(
        (path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if not photos:
        raise ValueError(f"No supported images found in {source_dir}")

    run_dir = next_run_dir(outputs)
    entries = []
    for number, photo in enumerate(photos, start=1):
        alias = f"image{number}{photo.suffix.lower()}"
        shutil.copy2(photo, run_dir / alias)
        entries.append({"source": photo.name, "image": alias, "status": "pending"})

    manifest = {"source_directory": str(source_dir.resolve()), "images": entries}
    save_manifest(run_dir, manifest)
    return run_dir, manifest


def post_image(url: str, image: Path, timeout: int) -> dict:
    boundary = f"wine-ocr-{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("ascii")
    body = head + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode("ascii")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_review(run_dir: Path, manifest: dict, url: str, timeout: int) -> bool:
    succeeded = True
    total = len(manifest["images"])
    for number, entry in enumerate(manifest["images"], start=1):
        stem = Path(entry["image"]).stem
        json_path = run_dir / f"{stem}_output.json"
        txt_path = run_dir / f"{stem}_output.txt"
        if json_path.exists() and txt_path.exists():
            print(f"[{number}/{total}] {stem}: already complete", flush=True)
            continue

        started = time.monotonic()
        try:
            payload = post_image(url, run_dir / entry["image"], timeout)
            result = payload["result"]
            report = payload["txt_report"]
            if not isinstance(result, dict) or not isinstance(report, str):
                raise ValueError("The OCR response has an unexpected structure")
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            txt_path.write_text(report, encoding="utf-8")
            entry["status"] = "ok"
            entry["elapsed_seconds"] = round(time.monotonic() - started, 2)
            entry["text_blocks"] = len(result.get("text_blocks", []))
            entry["candidate_fields"] = len(result.get("candidate_fields", []))
            print(
                f"[{number}/{total}] {stem}: OK, "
                f"{entry['text_blocks']} blocks, {entry['candidate_fields']} candidates, "
                f"{entry['elapsed_seconds']} s",
                flush=True,
            )
        except (HTTPError, URLError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            succeeded = False
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(f"[{number}/{total}] {stem}: ERROR: {exc}", file=sys.stderr, flush=True)
        finally:
            save_manifest(run_dir, manifest)
    return succeeded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "fixtures" / "manual_review" / "photos",
        help="Directory containing source photos (defaults to tests/fixtures/manual_review/photos)",
    )
    parser.add_argument("--outputs", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--url", default="http://127.0.0.1:8000/ocr")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--resume", type=Path, help="Resume an existing testN directory")
    args = parser.parse_args()

    if args.resume:
        run_dir = args.resume
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    else:
        run_dir, manifest = prepare_run(args.source_dir, args.outputs)
    print(f"Review directory: {run_dir.resolve()}", flush=True)
    return 0 if run_review(run_dir, manifest, args.url, args.timeout) else 1


if __name__ == "__main__":
    raise SystemExit(main())
