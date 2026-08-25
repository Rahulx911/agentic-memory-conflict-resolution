"""Cross-session recall scenarios — the other Phase 4 headline property besides
conflict resolution: a fact from an early session must still be findable many
sessions later.

Structured facts aren't session-scoped for reads (see
test_eval_metrics.py::test_structured_cross_session_recall_after_simulated_session_gap),
so that half of the property is close to true by construction. The half worth
testing live is episodic/semantic memory: a session-1 incident summary must
still rank as relevant when a semantically related question is asked in
session 5, after several unrelated sessions were embedded and upserted in
between.

Requires Qdrant up and a real VOYAGE_API_KEY (Voyage's free tier is 3 RPM, so
this is paced with sleeps between embed calls rather than firing them back to
back):
    .venv/bin/pytest tests/scenarios/test_cross_session_recall.py -v
"""

import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.memory.embeddings import embed_text
from src.memory.qdrant_store import (
    EpisodicSummary,
    _client,
    collection_name,
    init_collection,
    search,
    upsert_summary,
)

_VOYAGE_PACING_SECONDS = 21  # free tier is 3 RPM; stay comfortably under it


@pytest.fixture(autouse=True)
def _clean_collection():
    client = _client()
    name = collection_name()
    if client.collection_exists(name):
        client.delete_collection(name)
    init_collection()
    yield


def _write_episode(text: str, entity_names: list[str], occurred_at: datetime) -> uuid.UUID:
    session_id = uuid.uuid4()
    embedding = embed_text(text, input_type="document")
    upsert_summary(
        EpisodicSummary(session_id=session_id, summary=text, entity_names=entity_names, occurred_at=occurred_at),
        embedding,
    )
    return session_id


def test_session_1_incident_is_recalled_by_session_5_semantic_search():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    # Session 1: the incident that matters for this test.
    target_session_id = _write_episode(
        "Sensor_3 was damaged when a forklift collided with its housing in bay_1, "
        "cracking the enclosure and requiring an emergency replacement.",
        entity_names=["sensor_3", "bay_1"],
        occurred_at=t0,
    )
    time.sleep(_VOYAGE_PACING_SECONDS)

    # Sessions 2-4: unrelated topics, standing in for the sessions in between.
    unrelated_texts = [
        "Routine monthly inventory count completed in loading_dock with no discrepancies found.",
        "New warehouse floor staff completed onboarding and safety orientation this week.",
        "Zone bay_12's hazard signage was reinstalled after a scheduled inspection.",
    ]
    for i, text in enumerate(unrelated_texts):
        _write_episode(text, entity_names=[], occurred_at=t0 + timedelta(days=i + 1))
        time.sleep(_VOYAGE_PACING_SECONDS)

    # Session 5: a related question, phrased differently from the original report.
    query = "Has sensor_3 ever been struck or damaged by equipment before?"
    query_embedding = embed_text(query, input_type="query")
    results = search(query_embedding, limit=3)

    assert len(results) > 0
    top_session_ids = {r.payload.get("session_id") for r in results}
    assert str(target_session_id) in top_session_ids, [
        (r.payload.get("session_id"), r.payload.get("summary"), r.score) for r in results
    ]
