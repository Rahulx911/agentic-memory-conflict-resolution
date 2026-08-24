import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used-in-these-tests")

import pytest
from sqlalchemy import text

from src.memory.db import SessionLocal, engine, init_db


@pytest.fixture(scope="session", autouse=True)
def _init_schema():
    init_db()


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        # Tests write real rows against the dev Postgres; wipe between tests
        # so each test starts from a clean slate regardless of commit/rollback.
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE facts, escalations, entities, sessions CASCADE"))
