"""Resolve user-supplied wine facts before persisting a photo or reference."""
from __future__ import annotations

import math
import os
import re

from .feedback import normalize_slug
from .text_quality import words, GENERIC


class TextLabelMatcher:
    def __init__(self, client, catalog):
        self.client, self.catalog = client, catalog
        self.url = os.getenv("OCR_URL", "http://ocr:8000").rstrip("/")
        self.minimum = float(os.getenv("TEXT_LABEL_MIN_SCORE", ".90"))
        self.margin = float(os.getenv("TEXT_LABEL_MIN_MARGIN", ".015"))
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (self.minimum, self.margin)):
            raise ValueError("Text label thresholds must be between zero and one")

    async def resolve(self, information):
        values = information.model_dump()
        # Exact catalog identifiers remain supported for internal debugging;
        # the product form deliberately does not mention this capability.
        for value in values.values():
            if not isinstance(value, str):
                continue
            urls = re.findall(r"https://vino-svoe\.ru/wines/[a-z0-9][a-z0-9_-]*", value)
            if urls:
                slugs = {normalize_slug(url) for url in urls}
                if len(slugs) != 1:
                    return {"accepted": False, "reason": "conflicting_identifiers"}
                slug = slugs.pop()
                if slug not in await self.catalog.cards([slug]):
                    return {"accepted": False, "reason": "unknown_identifier"}
                return {"accepted": True, "slug": slug, "method": "catalog_identifier", "calibrated": False}
        text = "\n".join(str(value).strip() for value in values.values() if value is not None and str(value).strip())
        tokens = words(text) - GENERIC
        if len(tokens) < 2 or sum(map(len, tokens)) < 10:
            return {"accepted": False, "reason": "insufficient_information"}
        response = await self.client.post(self.url + "/match/text", json={"text": text})
        response.raise_for_status()
        matches = response.json()["top_10"]
        ranked = sorted(((str(identity), float(score)) for identity, score in matches.items()),
                        key=lambda item: -item[1])
        if not ranked:
            return {"accepted": False, "reason": "no_catalog_match"}
        if any(not math.isfinite(score) or not 0 <= score <= 1 for _, score in ranked):
            raise ValueError("Invalid text comparison score")
        score = ranked[0][1]
        gap = score - ranked[1][1] if len(ranked) > 1 else None
        diagnostics = {"method": "multilingual-e5-base", "calibrated": False,
                       "similarity": score, "margin": gap,
                       "thresholds": {"similarity": self.minimum, "margin": self.margin}}
        if score < self.minimum or gap is None or gap < self.margin:
            return {**diagnostics, "accepted": False, "reason": "ambiguous_text_match"}
        mapping = await self.catalog.slugs([ranked[0][0]])
        slug = mapping.get(ranked[0][0])
        if slug is None:
            raise ValueError("Text matcher returned unknown catalog ID")
        cards = await self.catalog.cards([slug])
        if information.year and cards[slug].get("vintage") not in {None, information.year}:
            return {**diagnostics, "accepted": False, "reason": "vintage_mismatch"}
        return {**diagnostics, "accepted": True, "slug": slug}
