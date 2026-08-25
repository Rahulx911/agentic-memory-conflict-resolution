"""Embedding client for episodic memory. Uses Voyage AI (Anthropic's recommended
embeddings provider — Claude itself has no embeddings endpoint).
"""

import os
import time

import voyageai
from voyageai.error import RateLimitError

MODEL = "voyage-3-lite"
EMBEDDING_DIM = 512  # voyage-3-lite output size; keep in sync with qdrant_store.EMBEDDING_DIM

# Voyage's free tier (no payment method on file) is 3 RPM. A single agent turn
# can trigger more than one embed call (retrieve_memory's query embed, plus an
# incident_search tool call embedding again), so a transient 429 there is an
# expected, recoverable condition, not a real failure — worth one paced retry
# rather than surfacing an error mid-conversation.
RATE_LIMIT_RETRIES = 2
RATE_LIMIT_BACKOFF_SECONDS = 21

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
    return _client


def embed_text(text: str, input_type: str = "document") -> list[float]:
    """input_type is 'document' when embedding content to store, 'query' when embedding a search query."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            result = _get_client().embed([text], model=MODEL, input_type=input_type)
            return result.embeddings[0]
        except RateLimitError:
            if attempt == RATE_LIMIT_RETRIES:
                raise
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
    raise AssertionError("unreachable")  # loop always returns or raises
