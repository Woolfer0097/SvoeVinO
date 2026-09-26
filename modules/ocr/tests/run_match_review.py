"""Run /match on review photos and save ranked IDs in a new outputs/testN."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

from run_manual_review import post_image, prepare_run, save_manifest


def validate_match_response(result: dict, expected_count: int | None) -> int:
    """Reject malformed rankings before saving a review artifact."""

    if not isinstance(result, dict) or set(result) != {"top_10"} or not isinstance(result["top_10"], dict):
        raise ValueError("Unexpected /match response structure")
    ranked = result["top_10"]
    if len(ranked) > 10 or (expected_count is not None and len(ranked) != expected_count):
        raise ValueError(f"Unexpected candidate count: {len(ranked)}")
    scores = list(ranked.values())
    if any(not isinstance(key, str) or not key.isdigit() for key in ranked):
        raise ValueError("Non-numeric wine ID")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > 1
        for value in scores
    ):
        raise ValueError("Invalid candidate score")
    if scores != sorted(scores, reverse=True):
        raise ValueError("Candidates are not sorted by score")
    return len(ranked)


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
    parser.add_argument("--url", default="http://127.0.0.1:8001/match")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--expected-count", type=int, choices=range(0, 11))
    parser.add_argument("--resume", type=Path, help="Resume an existing testN directory")
    args = parser.parse_args()

    if args.resume:
        run_dir = args.resume
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    else:
        run_dir, manifest = prepare_run(args.source_dir, args.outputs)
    print(f"Review directory: {run_dir}", flush=True)
    failed = False
    total = len(manifest["images"])
    for number, entry in enumerate(manifest["images"], start=1):
        stem = Path(entry["image"]).stem
        result_path = run_dir / f"{stem}_match_output.json"
        if result_path.exists():
            try:
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                validate_match_response(saved, args.expected_count)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            else:
                print(f"[{number}/{total}] {stem}: already complete", flush=True)
                continue
        started = time.monotonic()
        try:
            result = post_image(args.url, run_dir / entry["image"], args.timeout)
            count = validate_match_response(result, args.expected_count)
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            entry["status"] = "ok"
            entry["elapsed_seconds"] = round(time.monotonic() - started, 2)
            print(f"[{number}/{total}] {stem}: {count} candidates, {entry['elapsed_seconds']} s", flush=True)
        except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            failed = True
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(f"[{number}/{total}] {stem}: ERROR {exc}", flush=True)
        finally:
            save_manifest(run_dir, manifest)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
