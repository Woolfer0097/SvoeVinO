"""Conservative, inspectable OCR sufficiency heuristic for adaptive fusion."""
from __future__ import annotations

import math
import re

WORDS = re.compile(r"[a-zа-я]{3,}")
GENERIC = set("вино винодельня винодельни семейная сухое сухой полусухое полусладкое сладкое белое красное розовое игристое брют россия wine winery dry red white rose brut since founded рублей цена руб литр мл alcohol alc vol".split())


def words(text: str) -> set[str]:
    return set(WORDS.findall(text.casefold().replace("ё", "е")))


def assess_text(evidence: dict, cards: dict[str, dict], *, available: bool) -> dict:
    """Prices and producer-only text cannot identify a catalog variant.

    Confidence is from OCR recognition, not from E5 cosine similarity. This
    threshold is a starting heuristic to validate on the team's labeled data.
    """
    text = str(evidence.get("text") or "")
    tokens = words(text)
    producer = set().union(*(words(card.get("winery") or "") for card in cards.values()))
    informative = tokens - GENERIC - producer
    confidence = evidence.get("text_confidence")
    valid_confidence = (isinstance(confidence, (int, float))
                        and not isinstance(confidence, bool)
                        and math.isfinite(confidence) and 0 <= confidence <= 1)
    if not available:
        reason = "no_text_matches"
    elif len(tokens) < 3 or len(informative) < 2 or sum(map(len, informative)) < 12:
        reason = "too_little_distinctive_text"
    elif not valid_confidence:
        reason = "recognition_confidence_missing"
    elif confidence < .75:
        reason = "low_recognition_confidence"
    else:
        reason = "sufficient_text"
    return {"sufficient": reason == "sufficient_text", "reason": reason,
            "word_count": len(tokens), "informative_word_count": len(informative),
            "recognition_confidence": confidence if valid_confidence else None}
