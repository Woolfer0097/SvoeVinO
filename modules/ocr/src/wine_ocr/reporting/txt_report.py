"""Human-readable TXT report renderer."""

from __future__ import annotations

from collections import defaultdict

from ..contracts import CandidateField, OCRResult


def render_txt_report(result: OCRResult) -> str:
    """Render a human-readable OCR report."""

    lines: list[str] = []
    lines.append("Wine OCR Report")
    lines.append("================")
    lines.append("")
    lines.append(f"Engine: {result.engine}")
    lines.append(f"Processing time, ms: {result.processing_time_ms:.2f}")
    if result.warnings:
        lines.append(f"Warnings: {'; '.join(result.warnings)}")
    lines.append("")

    lines.append("Raw lines")
    lines.append("---------")
    if result.text_blocks:
        for block in result.text_blocks:
            confidence = _format_confidence(block.confidence)
            lines.append(
                f"- {block.text} [confidence: {confidence}, {block.source_variant}]"
            )
    else:
        lines.append("(no text recognized)")
    lines.append("")

    lines.append("Normalized text")
    lines.append("---------------")
    lines.append(result.normalized_text or "(empty)")
    lines.append("")

    grouped = _group_candidates(result.candidate_fields)
    _append_candidate_section(lines, "Candidate years", grouped.get("year", []))
    _append_candidate_section(
        lines,
        "Candidate percentages",
        grouped.get("percentage", []),
    )
    _append_candidate_section(lines, "Candidate volumes", grouped.get("volume", []))

    other_fields = [
        candidate
        for candidate in result.candidate_fields
        if candidate.field_type not in {"year", "percentage", "volume"}
    ]
    if other_fields:
        _append_candidate_section(lines, "Other candidates", other_fields)

    return "\n".join(lines).rstrip() + "\n"


def _group_candidates(
    candidates: list[CandidateField],
) -> dict[str, list[CandidateField]]:
    grouped: dict[str, list[CandidateField]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.field_type].append(candidate)
    return grouped


def _append_candidate_section(
    lines: list[str],
    title: str,
    candidates: list[CandidateField],
) -> None:
    lines.append(title)
    lines.append("-" * len(title))
    if not candidates:
        lines.append("(none)")
        lines.append("")
        return

    for candidate in candidates:
        confidence = _format_confidence(candidate.confidence)
        lines.append(
            f"- {candidate.normalized_value} "
            f"(raw: {candidate.value}, confidence: {confidence})"
        )
    lines.append("")


def _format_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "n/a"
    return f"{confidence:.3f}"
