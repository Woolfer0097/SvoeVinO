"""OpenCV-only diagnostic contact sheets for A-E variants."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from .models import ComparisonVariant, RGBArray

VARIANT_TITLES = {
    "A": "A original crop",
    "B": "B mild photometric",
    "C": "C cylindrical",
    "D": "D cylindrical + mild",
    "E": "E DewarpNet",
}


def _fit_image(image: RGBArray, width: int, height: int) -> RGBArray:
    scale = min(width / image.shape[1], height / image.shape[0])
    output_width = max(1, int(round(image.shape[1] * scale)))
    output_height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(
        image, (output_width, output_height), interpolation=interpolation
    )


def make_contact_sheet(
    variants: Mapping[str, ComparisonVariant],
    *,
    tile_width: int = 320,
    tile_height: int = 280,
) -> RGBArray:
    """Render fixed A-E tiles; missing outputs show their real reason."""

    if tile_width < 80 or tile_height < 80:
        raise ValueError("Contact-sheet tiles must be at least 80x80")
    header_height = 36
    content_height = tile_height - header_height
    sheet = np.full((tile_height, tile_width * 5, 3), 28, dtype=np.uint8)
    for index, code in enumerate("ABCDE"):
        x_start = index * tile_width
        tile = sheet[:, x_start : x_start + tile_width]
        cv2.putText(
            tile,
            VARIANT_TITLES[code],
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        variant = variants.get(code)
        if variant is not None and variant.image is not None:
            fitted = _fit_image(variant.image, tile_width, content_height)
            y = header_height + (content_height - fitted.shape[0]) // 2
            x = (tile_width - fitted.shape[1]) // 2
            tile[y : y + fitted.shape[0], x : x + fitted.shape[1]] = fitted
        else:
            reason = (
                variant.reason
                if variant is not None and variant.reason
                else "variant not available"
            )
            status = variant.status if variant is not None else "missing"
            cv2.putText(
                tile,
                status,
                (10, header_height + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 190, 70),
                1,
                cv2.LINE_AA,
            )
            words = reason.split()
            lines: list[str] = []
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if len(candidate) > 32 and current:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                lines.append(current)
            for line_index, line in enumerate(lines[:5]):
                cv2.putText(
                    tile,
                    line,
                    (10, header_height + 66 + line_index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (205, 205, 205),
                    1,
                    cv2.LINE_AA,
                )
        if index:
            sheet[:, x_start : x_start + 1] = 90
    return np.ascontiguousarray(sheet)

