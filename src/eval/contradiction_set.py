"""Labeled contradiction set for the conflict-resolution-accuracy metric.

Each case is a conflict scenario fed straight into `src.conflict.policy.decide`
(the same function `write_fact` calls in production) along with the label a
human reviewer would assign: should this auto-resolve, and if so, which side
should win. `conflict_resolution_accuracy` in `src.eval.metrics` scores the
policy's actual output against these labels.

Kept separate from `tests/test_conflict_policy.py`: that suite is a
correctness check on a handful of edge cases (frozen, must not silently
change). This set is a broader labeled sample meant to report a single
accuracy number and grow over time as new contradiction patterns show up.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

T0 = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class LabeledContradiction:
    case_id: str
    entity_type: str
    attribute: str
    candidate_confidence: float
    candidate_observed_at: datetime
    current_observed_at: datetime
    expected_auto_resolved: bool
    expected_new_is_current: bool | None  # None when expected_auto_resolved is False (n/a)


CONTRADICTION_SET: list[LabeledContradiction] = [
    LabeledContradiction(
        case_id="low_stakes_newer_wins",
        entity_type="equipment",
        attribute="last_serviced",
        candidate_confidence=0.9,
        candidate_observed_at=T0 + timedelta(days=2),
        current_observed_at=T0,
        expected_auto_resolved=True,
        expected_new_is_current=True,
    ),
    LabeledContradiction(
        case_id="low_stakes_backdated_loses",
        entity_type="equipment",
        attribute="last_serviced",
        candidate_confidence=0.9,
        candidate_observed_at=T0,
        current_observed_at=T0 + timedelta(days=2),
        expected_auto_resolved=True,
        expected_new_is_current=False,
    ),
    LabeledContradiction(
        case_id="zone_capacity_newer_wins",
        entity_type="zone",
        attribute="capacity",
        candidate_confidence=0.85,
        candidate_observed_at=T0 + timedelta(hours=6),
        current_observed_at=T0,
        expected_auto_resolved=True,
        expected_new_is_current=True,
    ),
    LabeledContradiction(
        case_id="equipment_status_never_auto_even_when_newer",
        entity_type="equipment",
        attribute="status",
        candidate_confidence=0.95,
        candidate_observed_at=T0 + timedelta(days=5),
        current_observed_at=T0,
        expected_auto_resolved=False,
        expected_new_is_current=None,
    ),
    LabeledContradiction(
        case_id="zone_hazard_never_auto",
        entity_type="zone",
        attribute="hazard",
        candidate_confidence=0.9,
        candidate_observed_at=T0 + timedelta(days=1),
        current_observed_at=T0,
        expected_auto_resolved=False,
        expected_new_is_current=None,
    ),
    LabeledContradiction(
        case_id="zone_status_never_auto",
        entity_type="zone",
        attribute="status",
        candidate_confidence=0.9,
        candidate_observed_at=T0 + timedelta(days=1),
        current_observed_at=T0,
        expected_auto_resolved=False,
        expected_new_is_current=None,
    ),
    LabeledContradiction(
        case_id="low_confidence_never_auto_even_if_newer",
        entity_type="equipment",
        attribute="last_serviced",
        candidate_confidence=0.3,
        candidate_observed_at=T0 + timedelta(days=3),
        current_observed_at=T0,
        expected_auto_resolved=False,
        expected_new_is_current=None,
    ),
    LabeledContradiction(
        case_id="exact_tie_falls_back_to_human",
        entity_type="equipment",
        attribute="last_serviced",
        candidate_confidence=0.9,
        candidate_observed_at=T0,
        current_observed_at=T0,
        expected_auto_resolved=False,
        expected_new_is_current=None,
    ),
    LabeledContradiction(
        case_id="borderline_confidence_at_threshold_is_low_stakes",
        entity_type="equipment",
        attribute="last_serviced",
        candidate_confidence=0.5,  # == LOW_CONFIDENCE_THRESHOLD, not below it
        candidate_observed_at=T0 + timedelta(days=1),
        current_observed_at=T0,
        expected_auto_resolved=True,
        expected_new_is_current=True,
    ),
    LabeledContradiction(
        case_id="unrecognized_entity_type_status_is_low_stakes",
        # "status" is only safety-relevant for equipment/zone; an entity type
        # outside that map isn't flagged just because the attribute is named "status".
        entity_type="user",
        attribute="status",
        candidate_confidence=0.9,
        candidate_observed_at=T0 + timedelta(days=1),
        current_observed_at=T0,
        expected_auto_resolved=True,
        expected_new_is_current=True,
    ),
]
