"""Find recognized fragments closest to annotated wine names in a saved run.

This is a text-only diagnostic. Similarity is not an accuracy score and cannot
prove whether a missing inscription was skipped by detection or misread by the
recognizer. Source photographs are never opened.
"""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from evaluate_manual_review import contains_phrase
from wine_ocr.postprocessing.normalization import normalize_text


def diagnose_image(expected_name: str, result: dict) -> dict:
    blocks = result["text_blocks"]
    options: list[dict] = []
    for variant in ("full", "central_crop"):
        indexed = [
            (index, block)
            for index, block in enumerate(blocks)
            if block["source_variant"] == variant
        ]
        for position, (index, block) in enumerate(indexed):
            options.append(_option(expected_name, variant, [index], [block]))
            if position + 1 < len(indexed):
                next_index, next_block = indexed[position + 1]
                options.append(
                    _option(
                        expected_name,
                        variant,
                        [index, next_index],
                        [block, next_block],
                    )
                )
    options.sort(key=lambda item: item["similarity"], reverse=True)
    return {
        "expected_name": expected_name,
        "candidate_name": result.get("candidate_name"),
        "name_in_normalized_text": contains_phrase(
            result["normalized_text"], expected_name
        ),
        "closest_fragments": options[:3],
    }


def _option(
    expected_name: str, variant: str, indexes: list[int], blocks: list[dict]
) -> dict:
    text = " ".join(block["text"] for block in blocks)
    expected = normalize_text(expected_name)
    observed = normalize_text(text)
    return {
        "text": text,
        "source_variant": variant,
        "block_indexes": indexes,
        "similarity": round(SequenceMatcher(None, expected, observed).ratio(), 3),
        "exact_phrase": contains_phrase(text, expected_name),
        "confidence": min(
            (block["confidence"] for block in blocks if block["confidence"] is not None),
            default=None,
        ),
    }


def diagnose_run(run_dir: Path, evaluation_path: Path | None = None) -> dict:
    evaluation_file = evaluation_path or run_dir / "evaluation.json"
    evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
    images = []
    for entry in evaluation["images"]:
        stem = Path(entry["image"]).stem
        payload = json.loads(
            (run_dir / f"{stem}_output.json").read_text(encoding="utf-8")
        )
        images.append(
            {
                "image": entry["image"],
                "source": entry["source"],
                "name_candidate_exact": entry["matches"]["name_candidate_exact"],
                **diagnose_image(entry["expected"]["name"], payload["result"]),
            }
        )
    return {
        "method": (
            "Closest one- or two-block text after NFKC/case/whitespace normalization; "
            "SequenceMatcher ratio is a diagnostic hint, not an OCR accuracy metric. "
            "No visual inspection or inference about unseen text detection."
        ),
        "images": images,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--evaluation", type=Path, help="Evaluation JSON (defaults to <run_dir>/evaluation.json)")
    parser.add_argument("--output", type=Path, help="Diagnostics JSON (defaults to <run_dir>/diagnostics.json)")
    args = parser.parse_args()
    report = diagnose_run(args.run_dir, args.evaluation)
    output_path = args.output or args.run_dir / "diagnostics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Diagnostics: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
