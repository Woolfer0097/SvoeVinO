"""Compare a manual OCR run with the selected text in a ground-truth file.

The ground-truth file is read only. Evaluation results are saved inside the
run directory so later OCR configurations can be compared on the same photos.
Matches are intentionally strict: this script does not guess visually similar
letters, transliterate text, or treat approximate words as correct OCR.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.postprocessing.normalization import normalize_text


ENTRY_RE = re.compile(r"image(\d+)\s*:\s*\{([^}]*)\}", re.DOTALL)
FIELD_RE = re.compile(r'(name|years|other)\s*:\s*"([^"]*)"')


def parse_ground_truth(content: str) -> dict[int, dict[str, object]]:
    """Parse the small, human-edited `imageN: { ... }` review format."""

    entries: dict[int, dict[str, object]] = {}
    for number_text, body in ENTRY_RE.findall(content):
        number = int(number_text)
        if number in entries:
            raise ValueError(f"Duplicate ground-truth entry: image{number}")
        fields = dict(FIELD_RE.findall(body))
        if not fields.get("name") or "other" not in fields:
            raise ValueError(f"image{number} needs name and other fields")
        entries[number] = {
            "name": fields["name"].strip(),
            "years": _split_list(fields.get("years", "")),
            "other": _split_list(fields["other"]),
        }
    if not entries:
        raise ValueError("No imageN ground-truth entries found")
    return entries


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def contains_phrase(text: str, phrase: str) -> bool:
    """Require a complete, contiguous phrase after conservative normalization."""

    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)"
    return re.search(pattern, normalize_text(text)) is not None


def evaluate_image(
    expected: dict[str, object],
    result: dict[str, object],
    manifest_entry: dict[str, object],
) -> dict[str, object]:
    """Evaluate one already-saved OCR result without running inference."""

    expected_name = str(expected["name"])
    expected_years = set(expected["years"])
    observed_years = {
        str(field["value"])
        for field in result["candidate_fields"]
        if field["field_type"] == "year"
    }
    normalized_text = str(result["normalized_text"])
    candidate_name = result.get("candidate_name")
    other_matches = [
        {"text": phrase, "in_normalized_text": contains_phrase(normalized_text, phrase)}
        for phrase in expected["other"]
    ]
    block_variants = Counter(
        str(block["source_variant"]) for block in result["text_blocks"]
    )

    return {
        "image": manifest_entry["image"],
        "source": manifest_entry["source"],
        "elapsed_seconds": manifest_entry.get("elapsed_seconds"),
        "expected": expected,
        "observed": {
            "candidate_name": candidate_name,
            "years": sorted(observed_years),
            "normalized_text": normalized_text,
            "retained_blocks_by_variant": dict(sorted(block_variants.items())),
            "variant_times_ms": result.get("variant_times_ms", {}),
        },
        "matches": {
            "name_candidate_exact": (
                normalize_text(str(candidate_name)) == normalize_text(expected_name)
                if candidate_name is not None
                else False
            ),
            "name_in_normalized_text": contains_phrase(normalized_text, expected_name),
            "years_found": sorted(expected_years & observed_years),
            "years_missing": sorted(expected_years - observed_years),
            "years_unexpected": sorted(observed_years - expected_years),
            "other": other_matches,
        },
    }


def evaluate_run(
    ground_truth: dict[int, dict[str, object]], run_dir: Path
) -> dict[str, object]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    images: list[dict[str, object]] = []
    for entry in manifest["images"]:
        match = re.fullmatch(r"image(\d+)\.[^.]+", str(entry["image"]))
        if match is None:
            raise ValueError(f"Unexpected image alias: {entry['image']}")
        number = int(match.group(1))
        if number not in ground_truth:
            raise ValueError(f"Missing ground truth for image{number}")
        output_path = run_dir / f"image{number}_output.json"
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        images.append(evaluate_image(ground_truth[number], payload["result"], entry))

    run_numbers = {
        int(re.fullmatch(r"image(\d+)\.[^.]+", str(e["image"])).group(1))
        for e in manifest["images"]
    }
    extra = sorted(set(ground_truth) - run_numbers)
    if extra:
        raise ValueError(f"Ground truth has no matching run images: {extra}")

    elapsed = [
        float(image["elapsed_seconds"])
        for image in images
        if image["elapsed_seconds"] is not None
    ]
    summary = {
        "images": len(images),
        "name_candidate_exact": sum(
            bool(image["matches"]["name_candidate_exact"]) for image in images
        ),
        "name_in_normalized_text": sum(
            bool(image["matches"]["name_in_normalized_text"]) for image in images
        ),
        "years_expected": sum(len(image["expected"]["years"]) for image in images),
        "years_found": sum(len(image["matches"]["years_found"]) for image in images),
        "years_unexpected": sum(
            len(image["matches"]["years_unexpected"]) for image in images
        ),
        "other_expected": sum(len(image["expected"]["other"]) for image in images),
        "other_found": sum(
            sum(bool(item["in_normalized_text"]) for item in image["matches"]["other"])
            for image in images
        ),
        "cold_seconds": elapsed[0] if elapsed else None,
        "warm_median_seconds": statistics.median(elapsed[1:]) if len(elapsed) > 1 else None,
    }
    return {"summary": summary, "images": images}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--label", default="", help="Human-readable OCR configuration label")
    parser.add_argument(
        "--output",
        type=Path,
        help="Evaluation JSON path (defaults to <run_dir>/evaluation.json)",
    )
    args = parser.parse_args()

    truth = parse_ground_truth(args.ground_truth.read_text(encoding="utf-8"))
    report = evaluate_run(truth, args.run_dir)
    report["ground_truth_file"] = str(args.ground_truth.resolve())
    report["run_directory"] = str(args.run_dir.resolve())
    report["label"] = args.label
    report["matching_rule"] = (
        "NFKC + casefold + whitespace normalization; whole contiguous phrase; "
        "no transliteration or visual-character substitutions"
    )
    output_path = args.output or args.run_dir / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Evaluation: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
