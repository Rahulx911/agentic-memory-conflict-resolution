import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used-in-these-tests")

import pytest
from sqlalchemy import text

from src.memory.db import SessionLocal, engine, init_db


@pytest.fixture(scope="session", autouse=True)
def _init_schema():
    init_db()


def _truncate() -> None:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE facts, escalations, entities, sessions CASCADE"))


@pytest.fixture
def db():
    # Truncate before, not just after: tests write real rows against the dev
    # Postgres, which is also used by manual runs of the CLIs and ad-hoc
    # scripts. Only cleaning up after each test leaves the *first* test in a
    # run at the mercy of whatever state that out-of-band usage left behind.
    _truncate()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        _truncate()
