"""Phase 4 evaluation harness: runs all four tracked metrics against the real
stack (Postgres, Qdrant, Claude, Voyage) and prints a report.

    .venv/bin/python -m src.eval.run_eval

Conflict-resolution accuracy and, for the seeded fixtures, retrieval
precision are cheap and deterministic. Cross-session recall and latency make
real Claude/Voyage calls, so this takes on the order of a couple of minutes
(paced to stay under Voyage's free-tier 3 RPM limit) rather than being
something to run on every commit — that's what the fast, mocked pytest suite
is for.
"""

import time

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages

from src.agent.graph import build_graph
from src.eval.mention_set import FIXTURE_ENTITIES, MENTION_SET
from src.eval.metrics import (
    LatencyReport,
    RecallExpectation,
    conflict_resolution_accuracy,
    memory_retrieval_precision,
    structured_cross_session_recall,
)
from src.memory.db import get_session, init_db
from src.memory.qdrant_store import init_collection
from src.memory.repository import create_session, get_or_create_entity, write_fact

TURN_PACING_SECONDS = 15  # keep Voyage's per-turn query embed comfortably under 3 RPM

LATENCY_PROMPTS = [
    "What's the current status of sensor_3?",
    "Has anything unusual happened in bay_1 recently?",
    "Any incidents involving sensor_30 that I should know about?",
]


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_conflict_accuracy() -> None:
    _print_header("Conflict-resolution accuracy")
    report = conflict_resolution_accuracy()
    print(f"{report.correct}/{report.total} correct ({report.accuracy:.0%})")
    for r in report.mismatches:
        print(f"  MISMATCH {r.case_id}: expected auto_resolved={r.expected_auto_resolved}, "
              f"got {r.actual_auto_resolved}")


def run_retrieval_precision() -> None:
    _print_header("Memory retrieval precision (entity-mention detection)")
    db = get_session()
    try:
        for entity_type, name in FIXTURE_ENTITIES:
            get_or_create_entity(db, entity_type, name)
        db.commit()
    finally:
        db.close()

    report = memory_retrieval_precision(MENTION_SET)
    print(f"precision={report.precision:.0%}  recall={report.recall:.0%}  "
          f"exact_match_rate={report.exact_match_rate:.0%}")
    for r in report.results:
        if not r.correct:
            print(f"  MISMATCH {r.case_id}: expected={set(r.expected)} actual={set(r.actual)}")


def run_structured_cross_session_recall() -> None:
    _print_header("Cross-session recall (structured facts)")
    db = get_session()
    try:
        write_fact(
            db,
            entity_type="equipment",
            entity_name="sensor_3",
            attribute="status",
            value={"value": "faulty"},
            confidence=0.9,
            session_id=None,
        )
        db.commit()

        # Simulate intervening sessions writing unrelated facts.
        for i in range(3):
            write_fact(
                db,
                entity_type="equipment",
                entity_name=f"eval_unrelated_{i}",
                attribute="status",
                value={"value": "ok"},
                confidence=0.9,
                session_id=None,
            )
        db.commit()

        recall = structured_cross_session_recall(
            db,
            [RecallExpectation("equipment", "sensor_3", "status", {"value": "faulty"})],
        )
    finally:
        db.close()
    print(f"recall={recall:.0%} (session-1 fact recalled correctly after 3 intervening sessions)")
    print("Episodic/semantic cross-session recall is covered live in "
          "tests/scenarios/test_cross_session_recall.py (not re-run here to avoid "
          "burning Voyage's free-tier rate limit twice).")


def run_latency() -> None:
    _print_header("Latency (perceive -> respond, end to end)")
    init_db()
    init_collection()
    graph = build_graph()

    db = get_session()
    try:
        session = create_session(db, user_id="eval_harness")
        db.commit()
        session_id = session.id
    finally:
        db.close()

    state = {
        "session_id": session_id,
        "user_id": None,
        "messages": [],
        "mentioned_entities": [],
        "retrieved_facts": [],
        "retrieved_episodes": [],
        "candidate_fact_count": 0,
    }

    samples: list[float] = []
    for i, prompt in enumerate(LATENCY_PROMPTS):
        state["messages"].append(HumanMessage(content=prompt))
        start = time.monotonic()
        respond_elapsed = None
        # stream_mode="updates" yields each node's *delta*, not the merged state, so
        # "messages" (the only field with a non-default reducer) has to be merged by
        # hand with add_messages; every other field is a plain last-write-wins overwrite,
        # matching what the graph's own Pregel executor does internally.
        for chunk in graph.stream(state, stream_mode="updates"):
            node_name = next(iter(chunk))
            update = chunk[node_name] or {}  # LangGraph reports a no-op ({}) update as None
            if node_name == "respond":
                respond_elapsed = time.monotonic() - start
            if "messages" in update:
                state["messages"] = add_messages(state["messages"], update["messages"])
            for key, value in update.items():
                if key != "messages":
                    state[key] = value
        if respond_elapsed is not None:
            samples.append(respond_elapsed)
            print(f"  turn {i + 1}: {respond_elapsed:.2f}s  ({prompt!r})")
        if i < len(LATENCY_PROMPTS) - 1:
            time.sleep(TURN_PACING_SECONDS)

    report = LatencyReport(samples_seconds=samples)
    print(f"p50={report.p50:.2f}s  p95={report.p95:.2f}s  mean={report.mean:.2f}s  (n={len(samples)})")


def main() -> None:
    init_db()
    run_conflict_accuracy()
    run_retrieval_precision()
    run_structured_cross_session_recall()
    run_latency()
    print()


if __name__ == "__main__":
    main()
