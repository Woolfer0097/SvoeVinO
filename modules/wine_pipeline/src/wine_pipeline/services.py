"""Adapters for existing service contracts and row-ID to slug reconciliation."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import os
from urllib.parse import quote

import httpx
from .catalog import Catalog, apply_vintage
from .color import ColorComparator, blend
from .fusion import fuse, scores_by_id
from .text_quality import assess_text
from .decision import DecisionPolicy, prefer_confirmed_reference


class Pipeline:
    def __init__(self, client: httpx.AsyncClient, catalog: Catalog) -> None:
        self.client = client
        self.catalog = catalog
        self.dino = os.getenv("DINO_URL", "http://dinov2:8000").rstrip("/")
        self.verifier = os.getenv("SUPERPOINT_URL", "http://superpoint:8000").rstrip("/")
        self.ocr = os.getenv("OCR_URL", "http://ocr:8000").rstrip("/")
        self.preprocessing = os.getenv("PREPROCESS_URL", "http://preprocessing:8000").rstrip("/")
        self.color_weight = float(os.getenv("COLOR_WEIGHT", ".05"))
        if not math.isfinite(self.color_weight) or not 0 <= self.color_weight <= .1:
            raise ValueError("COLOR_WEIGHT must be between zero and 0.1")
        self.colors = ColorComparator(client, self.dino)
        self.decision_policy = DecisionPolicy()

    async def _post(self, url: str, data: bytes, filename: str, field: str, form=None):
        response = await self.client.post(
            url, files={field: (filename, data, "application/octet-stream")}, data=form,
        )
        response.raise_for_status()
        return response.json()

    async def run(self, data: bytes, filename: str, progress):
        progress("preprocess", .1)
        prepared = await self._post(self.preprocessing + "/preprocess", data, filename, "image")
        try:
            visual_data = base64.b64decode(prepared["embedding_png"], validate=True)
            ocr_data = base64.b64decode(prepared["ocr_png"], validate=True)
            if not visual_data or not ocr_data or max(len(visual_data), len(ocr_data)) > 15 * 1024 * 1024:
                raise ValueError("Invalid preprocessing image size")
            metadata = prepared["metadata"]
            if not isinstance(metadata, dict):
                raise ValueError("Invalid preprocessing metadata")
        except (KeyError, TypeError, binascii.Error) as exc:
            raise ValueError("Invalid preprocessing response") from exc
        progress("preprocess_done", .2)
        filename = "prepared.png"

        async def visual_branch():
            progress("dino_started", .2)
            response = await self._post(
                self.dino + "/search", visual_data, filename, "file", {"top_k": "20"},
            )
            candidates = response["candidates"]
            if not candidates:
                raise ValueError("DINO returned no candidates; check reference publication")
            if (len(candidates) > 20
                    or len({c["slug"] for c in candidates}) != len(candidates)
                    or len({str(c["wine_id"]) for c in candidates}) != len(candidates)):
                raise ValueError("Invalid or duplicate DINO shortlist")
            for candidate in candidates:
                if not candidate["slug"] or not isinstance(candidate["slug"], str):
                    raise ValueError("DINO returned invalid slug")
                score = float(candidate["score"])
                if not math.isfinite(score) or not -1 <= score <= 1:
                    raise ValueError("DINO returned an invalid cosine similarity")
            progress("dino_done", .4)
            warnings = []
            try:
                progress("superpoint_started", .4)
                ids = [str(item["wine_id"]) for item in candidates]
                verified = await self._post(
                    self.verifier + "/verify", visual_data, filename, "query",
                    {"candidates": json.dumps(ids)},
                )
                scores = scores_by_id(verified, set(ids))
                mapping = {str(item["wine_id"]): item["slug"] for item in candidates}
                visual = {mapping[identity]: score for identity, score in scores.items()}
                progress("superpoint_done", .7)
            except Exception as exc:
                visual = {}
                warnings.append("superpoint_unavailable:" + type(exc).__name__)
                progress("superpoint_failed", .7)
            return response, candidates, visual, warnings

        async def text_branch():
            progress("ocr_started", .2)
            matched = await self._post(
                self.ocr + "/match?include_evidence=true", ocr_data, filename, "file",
            )
            scores = scores_by_id([
                {"id": identity, "score": score}
                for identity, score in matched["top_10"].items()
            ])
            mapping = await self.catalog.slugs(list(scores))
            if set(scores) - mapping.keys():
                raise ValueError("OCR returned unknown catalog IDs")
            by_slug = {}
            for identity, score in scores.items():
                slug = mapping[identity]
                by_slug[slug] = max(by_slug.get(slug, 0), score)
            evidence = matched.get("evidence", {})
            years = evidence.get("years", [])
            if not isinstance(years, list) or any(
                not isinstance(year, int) or not 1900 <= year <= 2099 for year in years
            ):
                raise ValueError("Invalid OCR year evidence")
            progress("ocr_done", .7)
            return by_slug, evidence

        progress("search", .2)
        visual_result, text_result = await asyncio.gather(
            visual_branch(), text_branch(), return_exceptions=True,
        )
        # DINO is the mandatory shortlist; other branches degrade explicitly.
        if isinstance(visual_result, BaseException):
            raise RuntimeError("Visual retrieval unavailable") from visual_result
        response, candidates, visual, warnings = visual_result
        if isinstance(text_result, BaseException):
            warnings.append("ocr_unavailable:" + type(text_result).__name__)
            text_result = ({}, {})
            progress("ocr_failed", .7)
        text_scores, evidence = text_result
        dino_count = len(candidates)
        dino_slugs = {candidate["slug"] for candidate in candidates}
        for candidate in candidates:
            candidate["retrieval_sources"] = ["dino"]
            if candidate["slug"] in text_scores:
                candidate["retrieval_sources"].append("ocr")
        extra = await self.catalog.references(
            [slug for slug in text_scores if slug not in dino_slugs], response["model_name"],
        )
        # Keep verifier's contract at <=20 IDs per call. The extra call is
        # bounded by OCR's Top-10; no unverified semantic hit becomes a fallback.
        if extra:
            progress("ocr_rescue_started", .7)
            try:
                extra_ids = [str(candidate["wine_id"]) for candidate in extra]
                verified = await self._post(
                    self.verifier + "/verify", visual_data, filename, "query",
                    {"candidates": json.dumps(extra_ids)},
                )
                scores = scores_by_id(verified, set(extra_ids))
                mapping = {str(candidate["wine_id"]): candidate["slug"] for candidate in extra}
                visual.update({mapping[identity]: score for identity, score in scores.items()})
                visual = dict(sorted(visual.items(), key=lambda item: -item[1])[:10])
                progress("ocr_rescue_done", .75)
            except Exception as exc:
                warnings.append("ocr_rescue_unavailable:" + type(exc).__name__)
                progress("ocr_rescue_failed", .75)
            candidates.extend(extra)
        progress("fusion", .8)
        cards = await self.catalog.cards([candidate["slug"] for candidate in candidates])
        for candidate in candidates:
            candidate.update(cards.get(candidate["slug"], {}))
        text_quality = assess_text(evidence, cards, available=bool(text_scores))
        ocr_weight = .7 if text_quality["sufficient"] else .3
        ranked, mode = fuse(candidates, visual, text_scores,
                            ocr_weight=ocr_weight, text_primary=text_quality["sufficient"])
        progress("color", .85)
        color_scores = await self.colors.compare(visual_data, ranked) if self.color_weight else {}
        ranked = blend(ranked, color_scores, self.color_weight)
        progress("vintage", .95)
        ranked, vintage_check = apply_vintage(ranked, evidence.get("years", []))
        if not ranked:
            raise RuntimeError("No wine prediction")
        ranked, confirmed_reference = prefer_confirmed_reference(ranked)
        if confirmed_reference:
            _, vintage_check = apply_vintage([ranked[0]], evidence.get("years", []))
        decision = self.decision_policy.decide(ranked, text_quality, vintage_check, confirmed_reference)
        for candidate in ranked:
            candidate["wine_url"] = (
                "https://vino-svoe.ru/wines/" + quote(candidate["slug"], safe="-_.~")
            )
        return {
            # Always return the best available candidate. Decision quality is
            # advisory, not a reason to hide a successful retrieval.
            "slug": ranked[0]["slug"], "status": "ok",
            "model_name": response["model_name"],
            "decision": decision,
            "message": None if decision["accepted"] else
                "Неуверенное совпадение: это самый близкий вариант из каталога. Сверьте название, рисунок этикетки и год.",
            "query_embedding_dimension": response["query_embedding_dimension"],
            "candidates": ranked, "fusion_mode": mode, "warnings": warnings,
            "fusion_weights": {"visual": round(1 - ocr_weight, 1), "ocr": ocr_weight},
            "text_quality": text_quality,
            "ocr_evidence": evidence,
            "pipeline_version": "adaptive-ocr-feedback-v3",
            "color_weight": self.color_weight if color_scores else 0,
            "vintage_check": vintage_check,
            "preprocessing": metadata,
            "branch_counts": {
                "dino": dino_count, "superpoint": len(visual), "ocr": len(text_scores),
                "ocr_rescue": len(extra),
            },
        }
