"""Pure unit tests for src.conflict.policy — no DB needed, these just check
the decision logic that write_fact() delegates to."""

from datetime import UTC, datetime, timedelta

from src.conflict.policy import LOW_CONFIDENCE_THRESHOLD, classify_stakes, decide

NOW = datetime.now(UTC)
EARLIER = NOW - timedelta(hours=1)
LATER = NOW + timedelta(hours=1)


def test_classify_stakes_safety_attribute_is_high():
    assert classify_stakes("equipment", "status", confidence=0.95) == "high"
    assert classify_stakes("zone", "hazard", confidence=0.95) == "high"


def test_classify_stakes_ordinary_attribute_is_low():
    assert classify_stakes("equipment", "last_serviced", confidence=0.95) == "low"


def test_classify_stakes_low_confidence_forces_high():
    assert classify_stakes("equipment", "last_serviced", confidence=LOW_CONFIDENCE_THRESHOLD - 0.01) == "high"


def test_decide_low_stakes_newer_observation_wins():
    decision = decide(
        entity_type="equipment",
        attribute="last_serviced",
        candidate_confidence=0.9,
        candidate_observed_at=LATER,
        current_observed_at=NOW,
    )
    assert decision.stakes == "low"
    assert decision.auto_resolved is True
    assert decision.new_is_current is True


def test_decide_low_stakes_older_observation_loses():
    """Out-of-order report: a low-stakes fact arriving with an earlier
    observed_at than what's already current must not overwrite it."""
    decision = decide(
        entity_type="equipment",
        attribute="last_serviced",
        candidate_confidence=0.9,
        candidate_observed_at=EARLIER,
        current_observed_at=NOW,
    )
    assert decision.stakes == "low"
    assert decision.auto_resolved is True
    assert decision.new_is_current is False


def test_decide_high_stakes_attribute_never_auto_resolves():
    decision = decide(
        entity_type="equipment",
        attribute="status",
        candidate_confidence=0.99,
        candidate_observed_at=LATER,
        current_observed_at=NOW,
    )
    assert decision.stakes == "high"
    assert decision.auto_resolved is False
    assert decision.new_is_current is False


def test_decide_low_confidence_never_auto_resolves_even_if_newer():
    decision = decide(
        entity_type="equipment",
        attribute="last_serviced",
        candidate_confidence=0.2,
        candidate_observed_at=LATER,
        current_observed_at=NOW,
    )
    assert decision.stakes == "high"
    assert decision.auto_resolved is False


def test_decide_tie_falls_back_to_human():
    decision = decide(
        entity_type="equipment",
        attribute="last_serviced",
        candidate_confidence=0.9,
        candidate_observed_at=NOW,
        current_observed_at=NOW,
    )
    assert decision.stakes == "high"
    assert decision.auto_resolved is False
