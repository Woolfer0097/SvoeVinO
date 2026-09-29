"""Conservative abstention rules, explicitly not calibrated probabilities."""
from __future__ import annotations

import math
import os


def prefer_confirmed_reference(ranked):
    exact = [item for item in ranked
             if str(item.get("best_image_uri", "")).startswith("/data/reference/feedback/")
             and (item.get("dino_score") or 0) >= .985
             and (item.get("visual_score") or 0) >= .95]
    if len(exact) != 1:
        return ranked, False
    chosen = dict(exact[0], selection_override="confirmed_reference")
    return [chosen, *(item for item in ranked if item["slug"] != chosen["slug"])], True


class DecisionPolicy:
    def __init__(self):
        self.minimum_score = self._threshold("MATCH_MIN_SCORE", .60)
        self.minimum_margin = self._threshold("MATCH_MIN_MARGIN", .03)
        self.minimum_geometry = self._threshold("MATCH_MIN_GEOMETRY", .65)

    @staticmethod
    def _threshold(name, default):
        value = float(os.getenv(name, str(default)))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")
        return value

    def decide(self, ranked, text_quality, vintage_check, confirmed_reference=False):
        top = ranked[0]
        score = float(top["score"])
        alternatives = ranked[1:]
        if vintage_check.get("state") == "matched":
            year = vintage_check["detected_year"]
            alternatives = [item for item in alternatives if item.get("vintage") in {None, year}]
        margin = score - alternatives[0]["score"] if alternatives else None
        geometry = top.get("visual_score") or 0
        dino = top.get("dino_score") or 0
        # Support from independent image signals, rather than E5 cosine alone.
        supported = (geometry >= self.minimum_geometry or dino >= .985 or
                     (text_quality["sufficient"] and geometry >= .35 and dino >= .70))
        if confirmed_reference:
            reason = "confirmed_reference"
        elif vintage_check.get("state") == "mismatch":
            reason = "vintage_mismatch"
        elif score < self.minimum_score:
            reason = "weak_score"
        elif margin is not None and margin < self.minimum_margin:
            reason = "ambiguous_candidates"
        elif not supported:
            reason = "insufficient_visual_support"
        else:
            reason = "supported_match"
        return {
            "accepted": reason in {"supported_match", "confirmed_reference"},
            "reason": reason, "calibrated": False, "policy_version": "conservative-v1",
            "score": score, "margin": margin, "visual_score": top.get("visual_score"),
            "dino_score": top.get("dino_score"), "ocr_score": top.get("ocr_score"),
            "thresholds": {"score": self.minimum_score, "margin": self.minimum_margin,
                           "geometry": self.minimum_geometry},
        }
