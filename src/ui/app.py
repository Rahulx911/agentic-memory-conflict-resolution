"""Minimal chat UI with a memory-state transparency sidebar: what the agent
retrieved this turn (current facts + related past-session episodes), and the
live state of the conflict-resolution queue (pending conflicts, open
escalations) — so "what does the agent believe, and why" is never a mystery.
Pending conflicts can be accepted/rejected directly from the sidebar, same as
src.conflict.review_cli.

    .venv/bin/pip install -e ".[ui]"
    .venv/bin/streamlit run src/ui/app.py

Requires Postgres + Qdrant up and ANTHROPIC_API_KEY / VOYAGE_API_KEY set,
same as src.agent.run_cli.
"""

import sys
from pathlib import Path

# `streamlit run` executes this file directly and only puts its own directory
# (src/ui/) on sys.path, unlike every other entrypoint in this project, which
# runs via `python -m src....` from the repo root and gets that for free.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import build_graph
from src.agent.nodes import ai_text
from src.memory import repository
from src.memory.db import get_session, init_db
from src.memory.episodic import write_episode
from src.memory.models import Entity
from src.memory.qdrant_store import init_collection

st.set_page_config(page_title="Warehouse Ops Assistant", layout="wide")


def _new_agent_state(session_id) -> dict:
    return {
        "session_id": session_id,
        "user_id": None,
        "messages": [],
        "mentioned_entities": [],
        "retrieved_facts": [],
        "retrieved_episodes": [],
        "candidate_fact_count": 0,
    }


def _start_session() -> None:
    init_db()
    init_collection()
    st.session_state.graph = build_graph()

    db = get_session()
    try:
        session = repository.create_session(db, user_id=None)
        db.commit()
        session_id = session.id
    finally:
        db.close()

    st.session_state.session_id = session_id
    st.session_state.agent_state = _new_agent_state(session_id)
    st.session_state.entities_touched = set()
    st.session_state.ended = False
    st.session_state.last_summary = None


def _end_session(session_id, messages, entities_touched) -> str:
    """Close the given session, writing an episodic summary only if there's
    actual conversation content — matches src.agent.run_cli's guard, avoiding
    a wasted Claude/Voyage call (and a content-free episodic-memory entry)
    for a session nobody used. Always closes the session row, though: a
    session that's abandoned (e.g. via "Start new session") without this
    would stay open forever and its conversation would never be summarized."""
    if messages:
        summary = write_episode(session_id, messages, list(entities_touched))
    else:
        summary = "(session ended with no messages exchanged)"
    db = get_session()
    try:
        repository.close_session(db, session_id, summary)
        db.commit()
    finally:
        db.close()
    return summary


if "session_id" not in st.session_state:
    _start_session()


def _entity_label(db, entity_id) -> str:
    entity = db.get(Entity, entity_id)
    return f"{entity.entity_type}:{entity.name}" if entity else str(entity_id)


def _render_conflict_queue() -> None:
    st.subheader("Conflict queue")
    db = get_session()
    try:
        pending = repository.list_pending_conflicts(db)
        escalations = repository.list_open_escalations(db)

        col1, col2 = st.columns(2)
        col1.metric("Pending conflicts", len(pending))
        col2.metric("Open escalations", len(escalations))

        for fact in pending:
            label = _entity_label(db, fact.entity_id)
            current = repository.get_current_fact(db, fact.entity_id, fact.attribute)
            with st.expander(f"{label}.{fact.attribute}"):
                st.markdown(f"**New:** `{fact.value}` (confidence {fact.confidence}, observed {fact.observed_at})")
                if current is not None:
                    st.markdown(
                        f"**Current:** `{current.value}` (confidence {current.confidence}, "
                        f"observed {current.observed_at})"
                    )
                st.caption(fact.resolution_reason)
                c1, c2 = st.columns(2)
                if c1.button("Accept new", key=f"accept-{fact.id}"):
                    repository.confirm_conflict(db, fact.id, accept=True)
                    db.commit()
                    st.rerun()
                if c2.button("Reject (keep current)", key=f"reject-{fact.id}"):
                    repository.confirm_conflict(db, fact.id, accept=False)
                    db.commit()
                    st.rerun()

        if escalations:
            st.markdown("**Open escalations:**")
            for esc in escalations:
                st.markdown(f"- {esc.reason}")
    finally:
        db.close()


with st.sidebar:
    st.header("Memory state")
    st.caption(f"Session `{str(st.session_state.session_id)[:8]}`")

    agent_state = st.session_state.agent_state

    st.subheader("Mentioned this turn")
    if agent_state["mentioned_entities"]:
        for e in agent_state["mentioned_entities"]:
            st.markdown(f"- **{e['entity_type']}**: {e['name']}")
    else:
        st.caption("No entities recognized in the last message.")

    st.subheader("Retrieved current facts")
    if agent_state["retrieved_facts"]:
        for f in agent_state["retrieved_facts"]:
            st.markdown(
                f"- **{f['entity_name']}.{f['attribute']}** = `{f['value']}`  \n"
                f"  confidence {f['confidence']:.2f}, observed {f['observed_at']}"
            )
    else:
        st.caption("Nothing retrieved yet.")

    st.subheader("Related past sessions")
    if agent_state["retrieved_episodes"]:
        for e in agent_state["retrieved_episodes"]:
            st.markdown(f"- ({e['score']:.2f}) {e['summary']}")
    else:
        st.caption("No related past sessions surfaced.")

    st.divider()
    _render_conflict_queue()

    st.divider()
    if not st.session_state.ended and st.button("End session", key="end_session_btn"):
        with st.spinner("Summarizing session..."):
            summary = _end_session(
                st.session_state.session_id, agent_state["messages"], st.session_state.entities_touched
            )
        st.session_state.ended = True
        st.session_state.last_summary = summary
        st.rerun()
    if st.button("Start new session", key="start_new_session_btn"):
        if not st.session_state.ended:
            with st.spinner("Saving current session..."):
                _end_session(
                    st.session_state.session_id, agent_state["messages"], st.session_state.entities_touched
                )
        _start_session()
        st.rerun()


st.title("Warehouse Ops Assistant")
st.caption("Persistent memory across sessions, with explicit conflict resolution.")

for m in agent_state["messages"]:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.write(m.content)
    elif isinstance(m, AIMessage):
        text = ai_text(m)
        if text:
            with st.chat_message("assistant"):
                st.write(text)
        for tool_call in m.tool_calls:
            st.caption(f"🔧 called `{tool_call['name']}({tool_call.get('args', {})})`")

if st.session_state.ended:
    st.info(f"Session closed. Summary written to episodic memory:\n\n{st.session_state.last_summary}")
else:
    prompt = st.chat_input("Ask about equipment, zones, or report an incident...")
    if prompt:
        agent_state["messages"].append(HumanMessage(content=prompt))
        with st.spinner("Thinking..."):
            new_state = st.session_state.graph.invoke(agent_state)
        st.session_state.agent_state = new_state
        st.session_state.entities_touched.update(ref["name"] for ref in new_state["mentioned_entities"])
        st.rerun()
