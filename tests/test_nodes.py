"""Tests for src/agent/nodes.py's text-extraction helpers.

ChatAnthropic's AIMessage.content is a plain string only when there's no
tool use or extended thinking involved; the moment either is present it
becomes a list of typed content blocks (text/thinking/tool_use). Reading
`.content` directly (e.g. via `str(...)`) on such a message dumps raw JSON —
including tool_use args and a thinking signature — instead of the model's
actual reply text. This matters most for extract_memory: it feeds
_current_turn_ai_text's output straight to a second LLM call that decides
what facts to write, so noise there can suppress a real extraction (found
live: a turn where the agent called a tool and then hedged about a conflict
produced zero extracted facts, because the "exchange" text handed to the
extractor was mostly raw tool-call JSON, not the agent's actual words).
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.nodes import _current_turn_ai_text, ai_text


def test_ai_text_returns_plain_string_as_is():
    assert ai_text(AIMessage(content="plain text reply")) == "plain text reply"


def test_ai_text_extracts_only_text_blocks_not_tool_use_or_thinking():
    message = AIMessage(
        content=[
            {"type": "thinking", "thinking": "", "signature": "deadbeef"},
            {"type": "tool_use", "id": "1", "name": "db_lookup", "input": {"entity_name": "sensor_3"}},
            {"type": "text", "text": "Here's what I found."},
        ]
    )
    assert ai_text(message) == "Here's what I found."


def test_ai_text_empty_when_only_tool_use_no_text_yet():
    message = AIMessage(content=[{"type": "tool_use", "id": "1", "name": "db_lookup", "input": {}}])
    assert ai_text(message) == ""


def test_current_turn_ai_text_joins_only_text_across_tool_calling_turn():
    state = {
        "messages": [
            HumanMessage(content="report something"),
            AIMessage(
                content=[{"type": "tool_use", "id": "1", "name": "db_lookup", "input": {"entity_name": "sensor_3"}}],
                tool_calls=[{"name": "db_lookup", "args": {"entity_name": "sensor_3"}, "id": "1", "type": "tool_call"}],
            ),
            ToolMessage(content="lookup result", tool_call_id="1"),
            AIMessage(content=[{"type": "text", "text": "Final answer here."}]),
        ]
    }
    text = _current_turn_ai_text(state)
    assert text == "Final answer here."
    # Regression guard: a naive str(m.content) would have leaked these.
    assert "tool_use" not in text
    assert "db_lookup" not in text
