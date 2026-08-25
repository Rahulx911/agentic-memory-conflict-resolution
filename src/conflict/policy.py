"""Conflict resolution policy: given a new observation that disagrees with the
current fact, decide whether it can be auto-resolved or must be flagged for a
human.

The policy is intentionally simple and legible rather than learned, because
the differentiating property this project targets is auditability: anyone
should be able to read `classify_stakes` and know exactly why a given
contradiction was or wasn't auto-resolved.

Two axes decide "stakes":
  - the attribute itself (e.g. equipment status is safety-relevant; a service
    date typo is not)
  - the new observation's own confidence (a low-confidence claim shouldn't be
    allowed to auto-overwrite anything, regardless of attribute)

Low-stakes conflicts are auto-resolved by recency: whichever observation has
the later `observed_at` wins. Ties can't be broken by recency, so they fall
back to a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

HIGH_STAKES_ATTRIBUTES: dict[str, set[str]] = {
    "equipment": {"status"},
    "zone": {"status", "hazard"},
}

LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class ConflictDecision:
    stakes: str  # "low" | "high"
    auto_resolved: bool
    new_is_current: bool
    reason: str


def classify_stakes(entity_type: str, attribute: str, confidence: float) -> str:
    if attribute in HIGH_STAKES_ATTRIBUTES.get(entity_type, set()):
        return "high"
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "high"
    return "low"


def decide(
    *,
    entity_type: str,
    attribute: str,
    candidate_confidence: float,
    candidate_observed_at: datetime,
    current_observed_at: datetime,
) -> ConflictDecision:
    stakes = classify_stakes(entity_type, attribute, candidate_confidence)

    if stakes == "high":
        why = (
            f"'{attribute}' is a safety-relevant attribute for {entity_type}"
            if attribute in HIGH_STAKES_ATTRIBUTES.get(entity_type, set())
            else f"new observation confidence {candidate_confidence:.2f} is below the "
            f"auto-resolve threshold ({LOW_CONFIDENCE_THRESHOLD})"
        )
        return ConflictDecision(
            stakes="high",
            auto_resolved=False,
            new_is_current=False,
            reason=f"flagged for human confirmation: {why}",
        )

    if candidate_observed_at > current_observed_at:
        return ConflictDecision(
            stakes="low",
            auto_resolved=True,
            new_is_current=True,
            reason=(
                f"auto-resolved by recency: new observation ({candidate_observed_at.isoformat()}) "
                f"is more recent than the current fact ({current_observed_at.isoformat()})"
            ),
        )

    if candidate_observed_at < current_observed_at:
        return ConflictDecision(
            stakes="low",
            auto_resolved=True,
            new_is_current=False,
            reason=(
                f"auto-resolved by recency: new observation ({candidate_observed_at.isoformat()}) "
                f"predates the current fact ({current_observed_at.isoformat()}); current fact retained"
            ),
        )

    # Exact tie: recency can't break it, so don't guess.
    return ConflictDecision(
        stakes="high",
        auto_resolved=False,
        new_is_current=False,
        reason="flagged for human confirmation: observed_at tie with current fact, recency cannot decide",
    )
