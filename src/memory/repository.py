"""Access layer over the structured-memory tables.

`write_fact` is the one function that matters most: it decides, at write
time, whether a new observation is a plain insert, a no-op (unchanged value),
or a conflict. Conflicts are *staged*, not resolved — the new row is written
with is_current=False and resolution=PENDING_CONFIRMATION, and the prior
fact stays authoritative. Phase 3 adds the policy node that walks pending
rows and either auto-resolves them by recency or leaves them for a human.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from src.memory.models import Entity, Escalation, Fact, Resolution
from src.memory.models import Session as SessionModel


def get_or_create_entity(db: DBSession, entity_type: str, name: str) -> Entity:
    entity = db.execute(
        select(Entity).where(Entity.entity_type == entity_type, Entity.name == name)
    ).scalar_one_or_none()
    if entity is None:
        entity = Entity(entity_type=entity_type, name=name)
        db.add(entity)
        db.flush()
    return entity


def list_entity_names(db: DBSession) -> list[str]:
    return [row[0] for row in db.execute(select(Entity.name)).all()]


def get_current_fact(db: DBSession, entity_id: uuid.UUID, attribute: str) -> Fact | None:
    return db.execute(
        select(Fact).where(
            Fact.entity_id == entity_id,
            Fact.attribute == attribute,
            Fact.is_current.is_(True),
        )
    ).scalar_one_or_none()


def get_current_facts_for_entity(db: DBSession, entity_id: uuid.UUID) -> list[Fact]:
    return list(
        db.execute(
            select(Fact).where(Fact.entity_id == entity_id, Fact.is_current.is_(True))
        ).scalars()
    )


def write_fact(
    db: DBSession,
    *,
    entity_type: str,
    entity_name: str,
    attribute: str,
    value: dict,
    confidence: float,
    session_id: uuid.UUID | None,
    observed_at: datetime | None = None,
) -> Fact:
    entity = get_or_create_entity(db, entity_type, entity_name)
    observed_at = observed_at or datetime.now(UTC)
    current = get_current_fact(db, entity.id, attribute)

    if current is None:
        fact = Fact(
            entity_id=entity.id,
            attribute=attribute,
            value=value,
            confidence=confidence,
            observed_at=observed_at,
            is_current=True,
            source_session_id=session_id,
            resolution=Resolution.NO_CONFLICT,
        )
        db.add(fact)
        db.flush()
        return fact

    if current.value == value:
        # Same belief reported again — no new row, just a no-op.
        return current

    # Conflicting observation: stage it, don't overwrite. Phase 3's resolver
    # decides whether this auto-wins by recency or needs a human.
    fact = Fact(
        entity_id=entity.id,
        attribute=attribute,
        value=value,
        confidence=confidence,
        observed_at=observed_at,
        is_current=False,
        source_session_id=session_id,
        supersedes_fact_id=current.id,
        resolution=Resolution.PENDING_CONFIRMATION,
        resolution_reason="conflicts with current fact; awaiting Phase 3 conflict-resolution policy",
    )
    db.add(fact)
    db.flush()
    return fact


def list_pending_conflicts(db: DBSession) -> list[Fact]:
    return list(
        db.execute(
            select(Fact).where(Fact.resolution == Resolution.PENDING_CONFIRMATION)
        ).scalars()
    )


def create_session(db: DBSession, user_id: str | None) -> SessionModel:
    session = SessionModel(user_id=user_id)
    db.add(session)
    db.flush()
    return session


def close_session(db: DBSession, session_id: uuid.UUID, summary: str) -> None:
    session = db.get(SessionModel, session_id)
    if session is not None:
        session.ended_at = datetime.now(UTC)
        session.summary = summary


def log_escalation(
    db: DBSession,
    *,
    reason: str,
    session_id: uuid.UUID | None,
    entity_id: uuid.UUID | None,
) -> Escalation:
    escalation = Escalation(reason=reason, session_id=session_id, entity_id=entity_id)
    db.add(escalation)
    db.flush()
    return escalation
