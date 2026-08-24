"""Embedding client for episodic memory. Uses Voyage AI (Anthropic's recommended
embeddings provider — Claude itself has no embeddings endpoint).
"""

import os

import voyageai

MODEL = "voyage-3-lite"
EMBEDDING_DIM = 512  # voyage-3-lite output size; keep in sync with qdrant_store.EMBEDDING_DIM

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
    return _client


def embed_text(text: str, input_type: str = "document") -> list[float]:
    """input_type is 'document' when embedding content to store, 'query' when embedding a search query."""
    result = _get_client().embed([text], model=MODEL, input_type=input_type)
    return result.embeddings[0]
