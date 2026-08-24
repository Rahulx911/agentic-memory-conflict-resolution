import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.memory.models import Base


def _database_url() -> str:
    user = os.environ.get("POSTGRES_USER", "agent_memory")
    password = os.environ.get("POSTGRES_PASSWORD", "agent_memory")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "agent_memory")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


engine = create_engine(_database_url())
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    """Create all tables. Fine for local dev; swap for Alembic migrations before prod."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()
