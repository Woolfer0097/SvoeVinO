"""Verify both public endpoints using a TEAM image, not organizer queries."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


def check_complete(status: dict, expected: str | None) -> None:
    if status["state"] != "done":
        raise RuntimeError(f"Recognition failed: {status.get('error')}")
    result = status["result"]
    if expected is not None and result["slug"] != expected:
        raise RuntimeError(f"Unexpected slug: {result['slug']} (expected {expected})")
    if result["warnings"]:
        raise RuntimeError(f"Degraded pipeline: {result['warnings']}")
    if result["branch_counts"]["dino"] == 0 or result["branch_counts"]["superpoint"] == 0:
        raise RuntimeError("Visual branch did not execute")
    # No readable text is valid, but cannot demonstrate a complete OCR/E5 match.
    if result["branch_counts"]["ocr"] == 0:
        raise RuntimeError("No OCR/E5 matches: choose a readable team validation image")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--expected-slug")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    content = args.image.read_bytes()
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": str(args.image), "expected_slug": args.expected_slug,
        "organizer_eval": False,
    }
    with httpx.Client(timeout=600) as client:
        start = time.monotonic()
        response = client.post(
            base + "/v1/eval/predict", files={"image": (args.image.name, content)},
        )
        response.raise_for_status()
        report["sync_seconds"] = round(time.monotonic() - start, 3)
        prediction = response.json()
        if not isinstance(prediction.get("slug"), str) or not prediction["slug"]:
            raise RuntimeError("Invalid organizer response")
        identity = response.headers["X-Job-ID"]
        status_response = client.get(base + "/image/status", params={"job_id": identity})
        status_response.raise_for_status()
        sync = status_response.json()
        check_complete(sync, args.expected_slug)
        assert prediction == {"slug": sync["result"]["slug"]}
        report["sync"] = sync

        start = time.monotonic()
        accepted = client.post(
            base + "/v1/eval/predict?wait=false",
            files={"image": (args.image.name, content)},
        )
        accepted.raise_for_status()
        if accepted.status_code != 202:
            raise RuntimeError("Async mode did not return HTTP 202")
        identity = accepted.json()["job_id"]
        states = []
        while time.monotonic() - start < 600:
            polled = client.get(base + "/image/status", params={"job_id": identity})
            polled.raise_for_status()
            status = polled.json()
            if status["state"] not in states:
                states.append(status["state"])
            if status["state"] in {"done", "failed"}:
                break
            time.sleep(.25)
        check_complete(status, args.expected_slug)
        report["async_seconds"] = round(time.monotonic() - start, 3)
        report["async_states"] = states
        report["async"] = status
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "slug": report["sync"]["result"]["slug"],
        "sync_seconds": report["sync_seconds"], "async_seconds": report["async_seconds"],
        "branch_counts": report["sync"]["result"]["branch_counts"],
        "fusion_mode": report["sync"]["result"]["fusion_mode"],
        "async_states": report["async_states"],
    }))


if __name__ == "__main__":
    main()
