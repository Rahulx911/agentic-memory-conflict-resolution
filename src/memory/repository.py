"""Access layer over the structured-memory tables.

`write_fact` is the one function that matters most: it decides, at write
time, whether a new observation is a plain insert, a no-op (unchanged value),
or a conflict. On conflict, `src.conflict.policy` decides whether it can be
auto-resolved by recency or must be staged as PENDING_CONFIRMATION and
escalated to a human — see that module for the policy itself.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session as DBSession

from src.conflict import policy
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


def list_entities(db: DBSession) -> list[Entity]:
    return list(db.execute(select(Entity)).scalars())


def find_entity_by_name(db: DBSession, name: str) -> Entity | None:
    """Case-insensitive lookup by name only (entities are unique per (type, name), so a
    name collision across types is possible but not handled here — first match wins).
    Pure read: never creates an entity, unlike get_or_create_entity."""
    return db.execute(
        select(Entity).where(func.lower(Entity.name) == name.lower())
    ).scalars().first()


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

    # Conflicting observation: ask the policy whether it auto-resolves by
    # recency or needs a human. Either way the prior fact is never deleted,
    # only its is_current flag flips — full history stays queryable.
    decision = policy.decide(
        entity_type=entity_type,
        attribute=attribute,
        candidate_confidence=confidence,
        candidate_observed_at=observed_at,
        current_observed_at=current.observed_at,
    )

    # The partial unique index on (entity_id, attribute) WHERE is_current is
    # not deferrable (Postgres can't defer a partial unique index), so the
    # old row must be flipped off — and flushed — before the new row is ever
    # inserted with is_current=True, or the two would transiently coexist
    # within the same flush and violate the constraint.
    if decision.new_is_current:
        current.is_current = False
        db.flush()

    fact = Fact(
        entity_id=entity.id,
        attribute=attribute,
        value=value,
        confidence=confidence,
        observed_at=observed_at,
        is_current=decision.new_is_current,
        source_session_id=session_id,
        supersedes_fact_id=current.id,
        resolution=Resolution.AUTO_RECENCY if decision.auto_resolved else Resolution.PENDING_CONFIRMATION,
        resolution_reason=decision.reason,
    )
    db.add(fact)
    db.flush()

    if not decision.auto_resolved:
        log_escalation(
            db,
            reason=f"conflict on {entity_type}:{entity_name}.{attribute} — {decision.reason}",
            session_id=session_id,
            entity_id=entity.id,
            fact_id=fact.id,
        )

    return fact


def confirm_conflict(db: DBSession, fact_id: uuid.UUID, *, accept: bool, reason: str | None = None) -> Fact:
    """Human resolution of a PENDING_CONFIRMATION fact. accept=True promotes the
    candidate to current (superseding whatever is *currently* current for that
    entity/attribute — not necessarily candidate.supersedes_fact_id, since another
    pending conflict on the same attribute may have already been resolved in the
    meantime); accept=False keeps the current fact and just closes out the
    candidate as rejected. Either way, any escalation raised for this specific
    fact is closed."""
    candidate = db.get(Fact, fact_id)
    if candidate is None or candidate.resolution != Resolution.PENDING_CONFIRMATION:
        raise ValueError(f"fact {fact_id} is not a pending conflict")

    candidate.resolution = Resolution.USER_CONFIRMED
    candidate.resolution_reason = reason or (
        "human operator confirmed this observation"
        if accept
        else "human operator rejected this observation; prior fact retained"
    )

    if accept:
        # Flip the current row off (and flush) before flipping the candidate
        # on — same non-deferrable-partial-index reasoning as in write_fact.
        current = get_current_fact(db, candidate.entity_id, candidate.attribute)
        if current is not None and current.id != candidate.id:
            current.is_current = False
            db.flush()
        candidate.is_current = True

    db.flush()

    for escalation in db.execute(
        select(Escalation).where(Escalation.fact_id == candidate.id, Escalation.status == "open")
    ).scalars():
        escalation.status = "resolved"

    return candidate


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
    fact_id: uuid.UUID | None = None,
) -> Escalation:
    escalation = Escalation(reason=reason, session_id=session_id, entity_id=entity_id, fact_id=fact_id)
    db.add(escalation)
    db.flush()
    return escalation


def list_open_escalations(db: DBSession) -> list[Escalation]:
    return list(db.execute(select(Escalation).where(Escalation.status == "open")).scalars())


def resolve_escalation(db: DBSession, escalation_id: uuid.UUID) -> None:
    escalation = db.get(Escalation, escalation_id)
    if escalation is not None:
        escalation.status = "resolved"
