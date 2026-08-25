"""Multi-session contradiction scenarios exercising the Phase 3 conflict
policy end-to-end through write_fact — the property that matters is that a
later session's report is resolved correctly against an earlier session's
report, with the right provenance trail, regardless of which session is
"newer" in wall-clock terms.

Run with Postgres up (docker compose -f docker/docker-compose.yml up -d):
    .venv/bin/pytest tests/scenarios -v
"""

from datetime import UTC, datetime, timedelta

from src.memory.models import Resolution
from src.memory.repository import (
    confirm_conflict,
    create_session,
    get_current_fact,
    get_or_create_entity,
    list_open_escalations,
    list_pending_conflicts,
    write_fact,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = T0 + timedelta(days=1)


def test_low_stakes_conflict_auto_resolves_by_recency_across_sessions(db):
    session_1 = create_session(db, user_id="ops_1")
    session_2 = create_session(db, user_id="ops_2")
    db.commit()

    write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="last_serviced",
        value={"value": "2026-01-01"},
        confidence=0.9,
        session_id=session_1.id,
        observed_at=T0,
    )
    db.commit()

    updated = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="last_serviced",
        value={"value": "2026-01-02"},
        confidence=0.9,
        session_id=session_2.id,
        observed_at=T1,
    )
    db.commit()

    assert updated.is_current is True
    assert updated.resolution == Resolution.AUTO_RECENCY
    assert updated.source_session_id == session_2.id

    entity = get_or_create_entity(db, "equipment", "sensor_3")
    db.commit()
    current = get_current_fact(db, entity.id, "last_serviced")
    assert current.id == updated.id
    assert list_pending_conflicts(db) == []
    assert list_open_escalations(db) == []


def test_low_stakes_backdated_report_does_not_overwrite_newer_current(db):
    """Session 2 runs after session 1 in wall-clock time, but reports an
    observation with an *earlier* observed_at (a backdated correction). It
    must not clobber the already-current, more-recently-observed fact."""
    session_1 = create_session(db, user_id="ops_1")
    session_2 = create_session(db, user_id="ops_2")
    db.commit()

    original = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="last_serviced",
        value={"value": "2026-01-02"},
        confidence=0.9,
        session_id=session_1.id,
        observed_at=T1,
    )
    db.commit()

    backdated = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="last_serviced",
        value={"value": "2026-01-01"},
        confidence=0.9,
        session_id=session_2.id,
        observed_at=T0,
    )
    db.commit()

    assert backdated.is_current is False
    assert backdated.resolution == Resolution.AUTO_RECENCY

    entity = get_or_create_entity(db, "equipment", "sensor_3")
    db.commit()
    current = get_current_fact(db, entity.id, "last_serviced")
    assert current.id == original.id
    assert list_pending_conflicts(db) == []


def test_high_stakes_conflict_across_sessions_flags_and_escalates_even_when_newer(db):
    session_1 = create_session(db, user_id="ops_1")
    session_2 = create_session(db, user_id="ops_2")
    db.commit()

    write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "faulty"},
        confidence=0.9,
        session_id=session_1.id,
        observed_at=T0,
    )
    db.commit()

    candidate = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "replaced"},
        confidence=0.9,
        session_id=session_2.id,
        observed_at=T1,  # more recent, but status is safety-relevant -> still flagged
    )
    db.commit()

    assert candidate.is_current is False
    assert candidate.resolution == Resolution.PENDING_CONFIRMATION

    entity = get_or_create_entity(db, "equipment", "sensor_3")
    db.commit()
    current = get_current_fact(db, entity.id, "status")
    assert current.value == {"value": "faulty"}  # unchanged, still authoritative

    pending = list_pending_conflicts(db)
    assert len(pending) == 1
    assert pending[0].id == candidate.id

    escalations = list_open_escalations(db)
    assert len(escalations) == 1
    assert escalations[0].entity_id == entity.id
    assert escalations[0].session_id == session_2.id


def test_human_accepts_pending_conflict_promotes_it_to_current(db):
    session_1 = create_session(db, user_id="ops_1")
    db.commit()

    write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "faulty"},
        confidence=0.9,
        session_id=session_1.id,
        observed_at=T0,
    )
    db.commit()
    candidate = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "replaced"},
        confidence=0.9,
        session_id=session_1.id,
        observed_at=T1,
    )
    db.commit()

    resolved = confirm_conflict(db, candidate.id, accept=True)
    db.commit()

    assert resolved.is_current is True
    assert resolved.resolution == Resolution.USER_CONFIRMED

    entity = get_or_create_entity(db, "equipment", "sensor_3")
    db.commit()
    current = get_current_fact(db, entity.id, "status")
    assert current.id == candidate.id
    assert current.value == {"value": "replaced"}
    assert list_pending_conflicts(db) == []


def test_human_rejects_pending_conflict_keeps_prior_current(db):
    session_1 = create_session(db, user_id="ops_1")
    db.commit()

    original = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "faulty"},
        confidence=0.9,
        session_id=session_1.id,
        observed_at=T0,
    )
    db.commit()
    candidate = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "replaced"},
        confidence=0.9,
        session_id=session_1.id,
        observed_at=T1,
    )
    db.commit()

    resolved = confirm_conflict(db, candidate.id, accept=False)
    db.commit()

    assert resolved.is_current is False
    assert resolved.resolution == Resolution.USER_CONFIRMED

    entity = get_or_create_entity(db, "equipment", "sensor_3")
    db.commit()
    current = get_current_fact(db, entity.id, "status")
    assert current.id == original.id
    assert list_pending_conflicts(db) == []
