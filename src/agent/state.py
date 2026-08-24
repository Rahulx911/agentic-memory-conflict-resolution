from __future__ import annotations

import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class EntityRef(TypedDict):
    id: uuid.UUID
    entity_type: str
    name: str


class RetrievedFact(TypedDict):
    entity_name: str
    attribute: str
    value: dict
    confidence: float
    observed_at: str


class RetrievedEpisode(TypedDict):
    session_id: str
    summary: str
    score: float


class AgentState(TypedDict):
    session_id: uuid.UUID
    user_id: str | None
    messages: Annotated[list[BaseMessage], add_messages]
    mentioned_entities: list[EntityRef]
    retrieved_facts: list[RetrievedFact]
    retrieved_episodes: list[RetrievedEpisode]
    candidate_fact_count: int
