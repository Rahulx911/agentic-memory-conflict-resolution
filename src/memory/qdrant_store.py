"""Episodic/semantic memory: per-session conversation summaries, embedded for
semantic recall across sessions. Structured facts live in Postgres (models.py);
this store answers "what happened before that's *like* this" rather than
"what do we currently believe."
"""

import os
import uuid
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBEDDING_DIM = 512  # voyage-3-lite output size; keep in sync with embeddings.EMBEDDING_DIM


def _client() -> QdrantClient:
    return QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
    )


def collection_name() -> str:
    return os.environ.get("QDRANT_COLLECTION", "episodic_memory")


def init_collection() -> None:
    client = _client()
    name = collection_name()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


@dataclass
class EpisodicSummary:
    session_id: uuid.UUID
    summary: str
    entity_names: list[str]
    occurred_at: datetime


def upsert_summary(summary: EpisodicSummary, embedding: list[float]) -> None:
    client = _client()
    client.upsert(
        collection_name=collection_name(),
        points=[
            PointStruct(
                id=str(summary.session_id),
                vector=embedding,
                payload={
                    "session_id": str(summary.session_id),
                    "summary": summary.summary,
                    "entity_names": summary.entity_names,
                    "occurred_at": summary.occurred_at.isoformat(),
                },
            )
        ],
    )


def search(embedding: list[float], limit: int = 5):
    client = _client()
    return client.query_points(
        collection_name=collection_name(),
        query=embedding,
        limit=limit,
    ).points
