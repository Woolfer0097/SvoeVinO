"""Generate a geometry-only cylindrical label example for the local CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def generate(output_directory: Path) -> tuple[Path, Path]:
    """Write a projected grid/text label and a benchmark JSONL manifest."""

    output_directory.mkdir(parents=True, exist_ok=True)
    height, width = 480, 640
    label_top, label_bottom = 140, 360
    cx, radius = 320.0, 230.0
    theta_min, theta_max = -0.85, 0.85
    flat_width = int(round(radius * (theta_max - theta_min)))
    flat_height = label_bottom - label_top

    flat = np.full((flat_height, flat_width, 3), (238, 226, 190), dtype=np.uint8)
    flat[:, ::32] = (80, 65, 45)
    flat[::28, :] = (125, 105, 75)
    cv2.putText(
        flat,
        "SYNTHETIC WINE",
        (22, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (35, 25, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        flat,
        "2024",
        (145, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (35, 25, 20),
        2,
        cv2.LINE_AA,
    )

    source = np.full((height, width, 3), (24, 28, 32), dtype=np.uint8)
    yy, xx = np.mgrid[:height, :width]
    ratio = (xx.astype(np.float32) - cx) / radius
    visible_body = np.abs(ratio) <= 1.0
    theta = np.arcsin(np.clip(ratio, -1.0, 1.0))
    map_u = ((theta - theta_min) / (theta_max - theta_min)) * (flat_width - 1)
    map_v = yy.astype(np.float32) - label_top
    projection = cv2.remap(
        flat,
        map_u.astype(np.float32),
        map_v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    label_visible = (
        visible_body
        & (theta >= theta_min)
        & (theta <= theta_max)
        & (yy >= label_top)
        & (yy < label_bottom)
    )
    source[label_visible] = projection[label_visible]

    image_path = output_directory / "synthetic_bottle.png"
    Image.fromarray(source, mode="RGB").save(image_path)
    label_x_min = int(round(cx + radius * np.sin(theta_min)))
    label_x_max = int(round(cx + radius * np.sin(theta_max)))
    manifest_path = output_directory / "benchmark.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_id": "synthetic-bottle",
                "image": image_path.name,
                "series_id": "synthetic-series-1",
                "split": "test",
                "conditions": ["synthetic", "strong_curvature"],
                "annotations": {
                    "label_bbox": [
                        label_x_min,
                        label_top,
                        label_x_max,
                        label_bottom,
                    ],
                    "cylinder": {
                        "cx": cx,
                        "radius": radius,
                        "theta_min": theta_min,
                        "theta_max": theta_max,
                    },
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return image_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    image_path, manifest_path = generate(args.output_directory)
    print(image_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
