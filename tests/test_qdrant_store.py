"""Tests for the Qdrant episodic store. Uses synthetic vectors so these don't
require a VOYAGE_API_KEY — the embedding provider is tested separately (it's a
thin wrapper around the Voyage SDK, not worth mocking).

Run with Qdrant up (docker compose -f docker/docker-compose.yml up -d):
    .venv/bin/pytest tests/test_qdrant_store.py -v
"""

import uuid
from datetime import UTC, datetime

import pytest

from src.memory.qdrant_store import (
    EMBEDDING_DIM,
    EpisodicSummary,
    init_collection,
    search,
    upsert_summary,
)


@pytest.fixture(scope="module", autouse=True)
def _init():
    init_collection()


def _vector(seed: float) -> list[float]:
    # A simple deterministic unit-ish vector so cosine similarity is meaningful.
    v = [0.0] * EMBEDDING_DIM
    v[0] = seed
    v[1] = 1.0 - seed
    return v


def test_upsert_and_exact_match_search():
    session_id = uuid.uuid4()
    summary = EpisodicSummary(
        session_id=session_id,
        summary="Operator reported sensor_3 as faulty during morning inspection.",
        entity_names=["sensor_3"],
        occurred_at=datetime.now(UTC),
    )
    vector = _vector(0.9)
    upsert_summary(summary, vector)

    results = search(vector, limit=5)
    assert any(r.payload["session_id"] == str(session_id) for r in results)
    top = next(r for r in results if r.payload["session_id"] == str(session_id))
    assert top.score > 0.99  # identical query vector should score near-perfect cosine similarity
    assert top.payload["summary"] == summary.summary
    assert top.payload["entity_names"] == ["sensor_3"]


def test_search_ranks_closer_vector_higher():
    near_id, far_id = uuid.uuid4(), uuid.uuid4()
    query_vector = _vector(0.8)

    upsert_summary(
        EpisodicSummary(near_id, "close match session", [], datetime.now(UTC)),
        _vector(0.8),
    )
    upsert_summary(
        EpisodicSummary(far_id, "far match session", [], datetime.now(UTC)),
        _vector(0.1),
    )

    results = search(query_vector, limit=10)
    ids_in_order = [r.payload["session_id"] for r in results]
    assert ids_in_order.index(str(near_id)) < ids_in_order.index(str(far_id))


def test_upsert_is_idempotent_on_session_id():
    session_id = uuid.uuid4()
    v1 = _vector(0.5)
    upsert_summary(EpisodicSummary(session_id, "first version", [], datetime.now(UTC)), v1)
    upsert_summary(EpisodicSummary(session_id, "updated version", [], datetime.now(UTC)), v1)

    results = search(v1, limit=10)
    matches = [r for r in results if r.payload["session_id"] == str(session_id)]
    assert len(matches) == 1
    assert matches[0].payload["summary"] == "updated version"
