"""Tests for the Phase 4 metrics harness in src/eval/metrics.py."""

from src.eval.mention_set import FIXTURE_ENTITIES, MENTION_SET
from src.eval.metrics import (
    LatencyReport,
    RecallExpectation,
    conflict_resolution_accuracy,
    memory_retrieval_precision,
    structured_cross_session_recall,
)
from src.memory.repository import get_or_create_entity, write_fact


def test_conflict_resolution_accuracy_is_perfect_against_the_labeled_set():
    """The labeled set encodes the policy's own documented behavior, so this
    is a regression guard: if it ever drops below 1.0, either the policy
    changed behavior or a label is stale — both worth knowing immediately."""
    report = conflict_resolution_accuracy()
    assert report.accuracy == 1.0, [r.case_id for r in report.mismatches]


def test_memory_retrieval_precision_catches_substring_false_positives(db):
    for entity_type, name in FIXTURE_ENTITIES:
        get_or_create_entity(db, entity_type, name)
    db.commit()

    report = memory_retrieval_precision(MENTION_SET)

    assert report.precision == 1.0, [
        (r.case_id, r.false_positives) for r in report.results if r.false_positives
    ]
    assert report.recall == 1.0, [
        (r.case_id, r.false_negatives) for r in report.results if r.false_negatives
    ]


def test_structured_cross_session_recall_after_simulated_session_gap(db):
    session_1 = get_or_create_entity(db, "equipment", "sensor_3")  # just to open a session-ish context
    db.commit()

    write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "faulty"},
        confidence=0.9,
        session_id=None,
    )
    db.commit()

    # Simulate several unrelated intervening sessions writing unrelated facts —
    # structured recall should be unaffected since facts aren't session-scoped.
    for i in range(4):
        write_fact(
            db,
            entity_type="equipment",
            entity_name=f"unrelated_{i}",
            attribute="status",
            value={"value": "ok"},
            confidence=0.9,
            session_id=None,
        )
    db.commit()

    recall = structured_cross_session_recall(
        db,
        [RecallExpectation("equipment", "sensor_3", "status", {"value": "faulty"})],
    )
    assert recall == 1.0
    assert session_1 is not None


def test_structured_cross_session_recall_reports_partial_when_value_diverged(db):
    write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "faulty"},
        confidence=0.9,
        session_id=None,
    )
    db.commit()

    recall = structured_cross_session_recall(
        db,
        [
            RecallExpectation("equipment", "sensor_3", "status", {"value": "faulty"}),
            RecallExpectation("equipment", "sensor_3", "status", {"value": "wrong_expected_value"}),
        ],
    )
    assert recall == 0.5


def test_structured_cross_session_recall_empty_expectations_is_vacuously_perfect(db):
    assert structured_cross_session_recall(db, []) == 1.0


def test_latency_report_percentiles():
    report = LatencyReport(samples_seconds=[float(i) for i in range(1, 101)])  # 1..100
    assert report.p50 == 50.0
    assert report.p95 == 95.0
    assert report.mean == 50.5


def test_latency_report_empty_samples_reports_zero():
    report = LatencyReport(samples_seconds=[])
    assert report.p50 == 0.0
    assert report.p95 == 0.0
    assert report.mean == 0.0
