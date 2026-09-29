"""Soft comparison of image palettes, not a classifier of wine type."""
from __future__ import annotations

import asyncio
import colorsys
import io
from collections import OrderedDict

from PIL import Image, ImageOps


def palette(data: bytes) -> list[float] | None:
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        # Conservative center/lower image proxy for a label, not a segmentation.
        width, height = image.size
        image = image.crop((int(width * .32), int(height * .4),
                            max(1, int(width * .68)), max(1, int(height * .85))))
        image.thumbnail((64, 96))
        bins = [0.0] * 14
        count = 0
        for red, green, blue, alpha in image.getdata():
            if alpha < 128:
                continue
            hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
            if value < .18:
                continue  # dark bottle glass should not dominate label color
            count += 1
            if saturation < .16:
                bins[13 if value > .7 else 12] += 1
            else:
                index = min(11, int(hue * 12))
                bins[index] += .6
                bins[(index - 1) % 12] += .2
                bins[(index + 1) % 12] += .2
        if count < 24:
            return None
        return [value / count for value in bins]


def similarity(left: list[float], right: list[float]) -> float:
    return max(0.0, min(1.0, sum(min(a, b) for a, b in zip(left, right))))


def blend(ranked: list[dict], scores: dict[str, float | None], weight: float) -> list[dict]:
    if not scores or weight == 0:
        return ranked
    for candidate in ranked:
        color = scores.get(candidate["slug"])
        candidate["fusion_score"] = candidate["score"]
        candidate["color_score"] = color
        # Missing reference color is neutral, not an automatic positive match.
        candidate["score"] = (1 - weight) * candidate["score"] + weight * (.5 if color is None else color)
    return sorted(ranked, key=lambda candidate: -candidate["score"])


class ColorComparator:
    def __init__(self, client, dino_url: str):
        self.client = client
        self.dino_url = dino_url
        self.cache = OrderedDict()
        self.semaphore = asyncio.Semaphore(4)

    async def compare(self, data: bytes, candidates: list[dict]) -> dict[str, float | None]:
        query = await asyncio.to_thread(palette, data)
        if query is None:
            return {}

        async def reference(candidate):
            uri = candidate.get("web_photo_uri") or candidate["best_image_uri"]
            try:
                if uri not in self.cache:
                    async with self.semaphore:
                        response = await self.client.get(
                            self.dino_url + "/images", params={"uri": uri, "max_side": 0},
                            timeout=15,
                        )
                        response.raise_for_status()
                        signature = await asyncio.to_thread(palette, response.content)
                    self.cache[uri] = signature
                    if len(self.cache) > 512:
                        self.cache.popitem(last=False)
                self.cache.move_to_end(uri)
                signature = self.cache[uri]
                return candidate["slug"], None if signature is None else similarity(query, signature)
            except Exception:
                return candidate["slug"], None

        results = dict(await asyncio.gather(*(reference(candidate) for candidate in candidates)))
        return results if any(score is not None for score in results.values()) else {}
