from __future__ import annotations

import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.state import AgentState
from src.memory import repository
from src.memory.db import get_session
from src.memory.embeddings import embed_text
from src.memory.qdrant_store import search as qdrant_search
from src.tools.tools import AGENT_TOOLS

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a warehouse-operations assistant with persistent, long-term memory.

You are given the facts and past-session context retrieved for this turn below. Treat
retrieved facts as your current beliefs, not certainties — if the user reports something
that contradicts a retrieved fact, do not silently pick a side; say so, and consider using
the escalate_to_human tool if the contradiction is safety-relevant (equipment status).

Use db_lookup for entities not already covered by retrieved facts, and incident_search
to check whether something similar has happened before.
"""


def _last_human_text(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def ai_text(message: AIMessage) -> str:
    """ChatAnthropic's content is either a plain string, or (whenever thinking/tool
    use is involved) a list of typed content blocks (text/thinking/tool_use) — only
    the text blocks are the model's actual reply. Anything that reads an AIMessage's
    content as prose (extraction, display) needs this, not raw `.content`."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _name_mentioned(name: str, text: str) -> bool:
    # Word-boundary match, not plain substring: entity names are underscore-joined
    # tokens (e.g. "bay_1", "bay_12"), and underscore counts as a word character in
    # regex, so \b correctly rejects "bay_1" matching inside "bay_12" while still
    # matching "bay_1" at a real token boundary (space, punctuation, string edge).
    return re.search(rf"\b{re.escape(name.lower())}\b", text) is not None


def perceive(state: AgentState) -> dict:
    text = _last_human_text(state).lower()
    db = get_session()
    try:
        entities = repository.list_entities(db)
        mentioned = [
            {"id": e.id, "entity_type": e.entity_type, "name": e.name}
            for e in entities
            if _name_mentioned(e.name, text)
        ]
    finally:
        db.close()
    return {"mentioned_entities": mentioned}


def retrieve_memory(state: AgentState) -> dict:
    retrieved_facts = []
    db = get_session()
    try:
        for ref in state["mentioned_entities"]:
            for f in repository.get_current_facts_for_entity(db, ref["id"]):
                retrieved_facts.append(
                    {
                        "entity_name": ref["name"],
                        "attribute": f.attribute,
                        "value": f.value,
                        "confidence": f.confidence,
                        "observed_at": f.observed_at.isoformat(),
                    }
                )
    finally:
        db.close()

    retrieved_episodes = []
    query_text = _last_human_text(state)
    if query_text:
        embedding = embed_text(query_text, input_type="query")
        for r in qdrant_search(embedding, limit=3):
            retrieved_episodes.append(
                {
                    "session_id": r.payload.get("session_id", ""),
                    "summary": r.payload.get("summary", ""),
                    "score": r.score,
                }
            )

    return {"retrieved_facts": retrieved_facts, "retrieved_episodes": retrieved_episodes}


def _format_context(state: AgentState) -> str:
    lines = [SYSTEM_PROMPT]
    if state["retrieved_facts"]:
        lines.append("\nRetrieved current facts:")
        for f in state["retrieved_facts"]:
            lines.append(f"- {f['entity_name']}.{f['attribute']} = {f['value']} (confidence {f['confidence']})")
    if state["retrieved_episodes"]:
        lines.append("\nRelated past sessions:")
        for e in state["retrieved_episodes"]:
            lines.append(f"- {e['summary']}")
    return "\n".join(lines)


_llm = ChatAnthropic(model=MODEL)
_llm_with_tools = _llm.bind_tools(AGENT_TOOLS)


def reason_act(state: AgentState) -> dict:
    system = SystemMessage(content=_format_context(state))
    response = _llm_with_tools.invoke([system, *state["messages"]])
    return {"messages": [response]}


def respond(state: AgentState) -> dict:
    # The final AI message is already in state["messages"] via the add_messages
    # reducer; this node exists as a named step to match the documented
    # perceive -> retrieve -> reason/act -> respond -> write architecture.
    return {}


class ExtractedFact(BaseModel):
    entity_type: str = Field(description="e.g. 'equipment' or 'zone'")
    entity_name: str
    attribute: str = Field(description="e.g. 'status', 'last_serviced'")
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


_extractor = ChatAnthropic(model=MODEL).with_structured_output(ExtractionResult)


def _current_turn_ai_text(state: AgentState) -> str:
    messages = state["messages"]
    last_human_idx = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    # Everything after the latest human turn, including intermediate tool-calling
    # AI messages, not just the final reply — a fact can be stated before the last tool call.
    ai_texts = [ai_text(m) for m in messages[last_human_idx + 1 :] if isinstance(m, AIMessage)]
    return " ".join(t for t in ai_texts if t)


def extract_memory(state: AgentState) -> dict:
    human = _last_human_text(state)
    exchange = f"User: {human}\nAgent: {_current_turn_ai_text(state)}"

    result: ExtractionResult = _extractor.invoke(
        [
            SystemMessage(
                content=(
                    "Extract any new facts about equipment or zones stated as true in this "
                    "exchange (status changes, service events, incident reports). Only extract "
                    "facts explicitly stated, not inferred. Return no facts if none were stated."
                )
            ),
            HumanMessage(content=exchange),
        ]
    )

    db = get_session()
    try:
        for f in result.facts:
            repository.write_fact(
                db,
                entity_type=f.entity_type,
                entity_name=f.entity_name,
                attribute=f.attribute,
                value={"value": f.value},
                confidence=f.confidence,
                session_id=state["session_id"],
            )
        db.commit()
    finally:
        db.close()

    return {"candidate_fact_count": len(result.facts)}
