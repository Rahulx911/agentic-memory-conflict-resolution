"""Headless smoke tests for the Streamlit chat UI (src/ui/app.py), using
Streamlit's AppTest framework. None of these send a chat message — that
would require a real Claude call, like the rest of the live scenarios in
this repo — so they're fast and free. They still exercise real Postgres +
Qdrant, real session creation, and the "Start new session" / "End session"
wiring, which is exactly what caught two real bugs during manual browser
testing: a session left open forever with no episodic summary when starting
a new one, and Claude/Voyage being called to summarize a session with no
messages in it.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used-in-these-tests")

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.memory.db import get_session
from src.memory.models import Session as SessionModel

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py")


def test_app_loads_without_error():
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception
    assert at.title[0].value == "Warehouse Ops Assistant"


def test_initial_sidebar_shows_empty_memory_state():
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception

    captions = [c.value for c in at.sidebar.caption]
    assert "No entities recognized in the last message." in captions
    assert "Nothing retrieved yet." in captions
    assert "No related past sessions surfaced." in captions

    metrics = {m.label: m.value for m in at.sidebar.metric}
    assert metrics["Pending conflicts"] == "0"
    assert metrics["Open escalations"] == "0"


def test_starting_a_new_session_closes_the_previous_one_without_a_real_llm_call(db):
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception
    first_session_id = at.session_state["session_id"]

    at.button(key="start_new_session_btn").click().run()

    assert not at.exception
    second_session_id = at.session_state["session_id"]
    assert second_session_id != first_session_id

    session_db = get_session()
    try:
        closed = session_db.get(SessionModel, first_session_id)
        assert closed is not None
        assert closed.ended_at is not None
        # A dummy ANTHROPIC_API_KEY is in play here, so if the empty-session
        # guard regressed and this tried to actually summarize via Claude, it
        # would either surface as at.exception (auth failure) or produce a
        # real (very different) summary — not this exact placeholder.
        assert closed.summary == "(session ended with no messages exchanged)"
    finally:
        session_db.close()


def test_ending_an_empty_session_marks_it_ended(db):
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception
    session_id = at.session_state["session_id"]

    at.button(key="end_session_btn").click().run()

    assert not at.exception
    assert at.session_state["ended"] is True
    assert "no messages exchanged" in at.session_state["last_summary"]

    session_db = get_session()
    try:
        closed = session_db.get(SessionModel, session_id)
        assert closed is not None
        assert closed.ended_at is not None
    finally:
        session_db.close()
