"""Tools available to the reason/act node: structured DB lookups, semantic
incident search over episodic memory, and human escalation.
"""


from langchain_core.tools import tool

from src.memory import repository
from src.memory.db import get_session
from src.memory.embeddings import embed_text
from src.memory.qdrant_store import search as qdrant_search


@tool
def db_lookup(entity_name: str) -> str:
    """Look up all current known facts for a named entity (equipment or zone) in structured memory."""
    db = get_session()
    try:
        entity = repository.get_or_create_entity(db, "equipment", entity_name)
        facts = repository.get_current_facts_for_entity(db, entity.id)
        db.rollback()  # get_or_create may have inserted a placeholder; don't persist a lookup miss
        if not facts:
            return f"No current facts on record for '{entity_name}'."
        lines = [f"- {f.attribute}: {f.value} (confidence {f.confidence}, observed {f.observed_at})" for f in facts]
        return f"Current facts for '{entity_name}':\n" + "\n".join(lines)
    finally:
        db.close()


@tool
def incident_search(query: str) -> str:
    """Semantically search past session summaries (episodic memory) for incidents or context related to the query."""
    embedding = embed_text(query, input_type="query")
    results = qdrant_search(embedding, limit=5)
    if not results:
        return "No related past sessions found."
    lines = [f"- ({r.payload.get('occurred_at')}) {r.payload.get('summary')}" for r in results]
    return "Related past sessions:\n" + "\n".join(lines)


@tool
def escalate_to_human(reason: str, entity_name: str | None = None) -> str:
    """Escalate an issue to a human operator when the agent cannot safely resolve it itself
    (e.g. a high-stakes fact conflict, or a request outside the agent's authority)."""
    db = get_session()
    try:
        entity_id = None
        if entity_name:
            entity = repository.get_or_create_entity(db, "equipment", entity_name)
            entity_id = entity.id
        repository.log_escalation(db, reason=reason, session_id=None, entity_id=entity_id)
        db.commit()
        return f"Escalated to human operator: {reason}"
    finally:
        db.close()


AGENT_TOOLS = [db_lookup, incident_search, escalate_to_human]
