"""Structured memory schema (Postgres).

Facts are append-only: a new observation for an (entity, attribute) pair does
not overwrite the previous row, it inserts a new one and flips `is_current`.
This makes the fact table its own provenance ledger — history is just
`SELECT * WHERE entity_id=... AND attribute=... ORDER BY created_at`.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Resolution(str, enum.Enum):
    NO_CONFLICT = "no_conflict"
    AUTO_RECENCY = "auto_recency"
    USER_CONFIRMED = "user_confirmed"
    PENDING_CONFIRMATION = "pending_confirmation"


class Session(Base):
    """One conversation session. Mirrors the episodic-memory point payload in Qdrant."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class Entity(Base):
    """Something the agent tracks facts about: equipment, a zone, a user, etc."""

    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("entity_type", "name", name="uq_entity_type_name"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "equipment", "zone"
    name: Mapped[str] = mapped_column(String(256), nullable=False)  # e.g. "sensor_3", "zone_b"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    facts: Mapped[list[Fact]] = relationship(back_populates="entity")


class Fact(Base):
    """One observation of an entity attribute at a point in time.

    Append-only. `is_current` marks the row currently believed true for
    (entity_id, attribute); superseded rows keep is_current=False rather than
    being deleted, so they remain queryable as history.
    """

    __tablename__ = "facts"
    __table_args__ = (
        # Application logic is responsible for only ever having one is_current
        # row per (entity, attribute), but a concurrent write or a resolver
        # bug could violate that silently — this makes it a hard DB
        # constraint instead, since "never silently overwrite" is the whole
        # point of this schema.
        Index(
            "uq_facts_current_per_entity_attribute",
            "entity_id",
            "attribute",
            unique=True,
            postgresql_where=text("is_current"),
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    entity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("entities.id"), nullable=False)
    attribute: Mapped[str] = mapped_column(String(128), nullable=False)  # e.g. "status", "last_serviced"
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    source_session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sessions.id"), nullable=True)
    supersedes_fact_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("facts.id"), nullable=True)
    resolution: Mapped[Resolution] = mapped_column(
        Enum(Resolution, name="resolution"), nullable=False, default=Resolution.NO_CONFLICT
    )
    resolution_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    entity: Mapped[Entity] = relationship(back_populates="facts")


class Escalation(Base):
    """A human-escalation request raised by the agent's escalate_to_human tool."""

    __tablename__ = "escalations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sessions.id"), nullable=True)
    entity_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("entities.id"), nullable=True)
    fact_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("facts.id"), nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")  # open | resolved
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
