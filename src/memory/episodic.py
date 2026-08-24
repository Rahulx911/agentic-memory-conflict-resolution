"""Writes episodic memory: summarize a finished session with Claude, embed the
summary, and upsert it to Qdrant for future semantic recall.
"""

import uuid
from datetime import UTC, datetime

import anthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.memory.embeddings import embed_text
from src.memory.qdrant_store import EpisodicSummary, upsert_summary

SUMMARY_MODEL = "claude-sonnet-5"


def _transcript(messages: list[BaseMessage]) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"Agent: {m.content}")
    return "\n".join(lines)


def summarize_session(messages: list[BaseMessage]) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this warehouse-ops conversation in 2-4 sentences, "
                    "naming any equipment/zones discussed and any facts or incidents "
                    "reported. This summary is stored as long-term memory for future "
                    "sessions, so be concrete and specific rather than vague.\n\n"
                    f"{_transcript(messages)}"
                ),
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def write_episode(session_id: uuid.UUID, messages: list[BaseMessage], entity_names: list[str]) -> str:
    summary = summarize_session(messages)
    embedding = embed_text(summary, input_type="document")
    upsert_summary(
        EpisodicSummary(
            session_id=session_id,
            summary=summary,
            entity_names=entity_names,
            occurred_at=datetime.now(UTC),
        ),
        embedding,
    )
    return summary
