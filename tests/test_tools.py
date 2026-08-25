"""Regression test for the escalate_to_human tool: it must attribute the
escalation it logs to the session that actually called it, via LangGraph's
InjectedState mechanism, rather than dropping session_id on the floor.

Drives the real compiled graph end-to-end but stubs out the LLM calls and the
Voyage/Qdrant calls, matching the rest of the suite's no-real-API-calls
convention.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.agent import nodes
from src.agent.graph import build_graph
from src.memory.repository import create_session, list_open_escalations


def test_escalate_to_human_tool_call_is_linked_to_the_calling_session(db, monkeypatch):
    session = create_session(db, user_id="ops_1")
    db.commit()

    tool_call = {
        "name": "escalate_to_human",
        "args": {"reason": "regression-test-escalation", "entity_name": None},
        "id": "call_1",
        "type": "tool_call",
    }
    responses = iter(
        [
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="Escalated."),
        ]
    )
    fake_llm_with_tools = type("_FakeLLM", (), {"invoke": staticmethod(lambda messages: next(responses))})()
    fake_extractor = type(
        "_FakeExtractor", (), {"invoke": staticmethod(lambda messages: nodes.ExtractionResult(facts=[]))}
    )()
    monkeypatch.setattr(nodes, "_llm_with_tools", fake_llm_with_tools)
    monkeypatch.setattr(nodes, "_extractor", fake_extractor)
    monkeypatch.setattr(nodes, "embed_text", lambda text, input_type="query": [0.0] * 512)
    monkeypatch.setattr(nodes, "qdrant_search", lambda embedding, limit=3: [])

    graph = build_graph()
    state = {
        "session_id": session.id,
        "user_id": None,
        "messages": [HumanMessage(content="please escalate this")],
        "mentioned_entities": [],
        "retrieved_facts": [],
        "retrieved_episodes": [],
        "candidate_fact_count": 0,
    }
    graph.invoke(state)

    escalations = list_open_escalations(db)
    matches = [e for e in escalations if e.reason == "regression-test-escalation"]
    assert len(matches) == 1
    assert matches[0].session_id == session.id
