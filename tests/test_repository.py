"""Tests for the conflict-staging behavior in src/memory/repository.py — the
core correctness property Phase 3 depends on: a contradictory observation
must never silently overwrite the current fact.

Run with Postgres up (docker compose -f docker/docker-compose.yml up -d):
    .venv/bin/pytest tests/test_repository.py -v
"""

import uuid

from src.memory.models import Resolution
from src.memory.repository import (
    close_session,
    create_session,
    find_entity_by_name,
    get_current_fact,
    get_current_facts_for_entity,
    get_or_create_entity,
    list_entities,
    list_pending_conflicts,
    log_escalation,
    write_fact,
)


def test_get_or_create_entity_is_idempotent(db):
    a = get_or_create_entity(db, "equipment", "sensor_3")
    b = get_or_create_entity(db, "equipment", "sensor_3")
    db.commit()
    assert a.id == b.id


def test_get_or_create_entity_distinguishes_by_type(db):
    equip = get_or_create_entity(db, "equipment", "bay_1")
    zone = get_or_create_entity(db, "zone", "bay_1")
    db.commit()
    assert equip.id != zone.id


def test_find_entity_by_name_case_insensitive(db):
    get_or_create_entity(db, "equipment", "Sensor_3")
    db.commit()
    assert find_entity_by_name(db, "sensor_3") is not None
    assert find_entity_by_name(db, "SENSOR_3") is not None
    assert find_entity_by_name(db, "nonexistent") is None


def test_write_fact_first_observation_is_current_no_conflict(db):
    fact = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "faulty"},
        confidence=0.9,
        session_id=None,
    )
    db.commit()
    assert fact.is_current is True
    assert fact.resolution == Resolution.NO_CONFLICT


def test_write_fact_same_value_is_a_noop(db):
    first = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "faulty"},
        confidence=0.9,
        session_id=None,
    )
    db.commit()

    second = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "faulty"},
        confidence=0.7,
        session_id=None,
    )
    db.commit()

    # Same fact id returned, no new row inserted for a repeated observation.
    assert second.id == first.id
    entity = get_or_create_entity(db, "equipment", "sensor_3")
    db.commit()
    assert len(get_current_facts_for_entity(db, entity.id)) == 1


def test_write_fact_conflict_is_staged_not_overwritten(db):
    """The critical property: a fact log saying 'replaced' followed by one saying
    'still faulty' must not blindly overwrite — the old fact stays current and
    the new one is staged as PENDING_CONFIRMATION for Phase 3 to resolve."""
    original = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "replaced"},
        confidence=0.9,
        session_id=None,
    )
    db.commit()

    conflicting = write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "faulty"},
        confidence=0.9,
        session_id=None,
    )
    db.commit()

    entity = get_or_create_entity(db, "equipment", "sensor_3")
    db.commit()

    # Old fact is still the one is_current=True — never silently overwritten.
    current = get_current_fact(db, entity.id, "status")
    assert current.id == original.id
    assert current.value == {"value": "replaced"}

    # New conflicting observation exists, but staged, not current.
    assert conflicting.is_current is False
    assert conflicting.resolution == Resolution.PENDING_CONFIRMATION
    assert conflicting.supersedes_fact_id == original.id

    pending = list_pending_conflicts(db)
    assert len(pending) == 1
    assert pending[0].id == conflicting.id


def test_write_fact_different_attributes_dont_conflict(db):
    write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="status",
        value={"value": "faulty"},
        confidence=0.9,
        session_id=None,
    )
    write_fact(
        db,
        entity_type="equipment",
        entity_name="sensor_3",
        attribute="last_serviced",
        value={"value": "2026-01-01"},
        confidence=0.9,
        session_id=None,
    )
    db.commit()
    assert list_pending_conflicts(db) == []


def test_list_entities_returns_all_types(db):
    get_or_create_entity(db, "equipment", "sensor_3")
    get_or_create_entity(db, "zone", "bay_1")
    db.commit()
    entities = list_entities(db)
    types = {e.entity_type for e in entities}
    assert types == {"equipment", "zone"}


def test_session_create_and_close(db):
    session = create_session(db, user_id="warehouse_ops_1")
    db.commit()
    assert session.ended_at is None

    close_session(db, session.id, summary="Discussed sensor_3 fault status.")
    db.commit()
    refreshed = db.get(type(session), session.id)
    assert refreshed.ended_at is not None
    assert refreshed.summary == "Discussed sensor_3 fault status."


def test_log_escalation_without_entity(db):
    escalation = log_escalation(db, reason="unrecognized equipment reported faulty", session_id=None, entity_id=None)
    db.commit()
    assert escalation.status == "open"
    assert escalation.entity_id is None


def test_log_escalation_linked_to_real_session_and_entity(db):
    session = create_session(db, user_id=None)
    entity = get_or_create_entity(db, "equipment", "sensor_3")
    db.commit()

    escalation = log_escalation(db, reason="conflicting fault reports", session_id=session.id, entity_id=entity.id)
    db.commit()

    assert isinstance(escalation.id, uuid.UUID)
    assert escalation.session_id == session.id
    assert escalation.entity_id == entity.id
