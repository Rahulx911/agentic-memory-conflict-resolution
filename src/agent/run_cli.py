"""Minimal REPL for exercising the agent graph end-to-end. Requires Postgres
and Qdrant running (see docker/docker-compose.yml) plus ANTHROPIC_API_KEY and
VOYAGE_API_KEY set.

    python -m src.agent.run_cli
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from src.agent.graph import build_graph
from src.memory.db import get_session, init_db
from src.memory.episodic import write_episode
from src.memory.qdrant_store import init_collection
from src.memory.repository import close_session, create_session


def main() -> None:
    init_db()
    init_collection()
    graph = build_graph()

    db = get_session()
    try:
        session = create_session(db, user_id=None)
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
    entities_touched: set[str] = set()

    print(f"Session {session_id} started. Ctrl-D or 'exit' to end.\n")
    try:
        while True:
            user_input = input("you> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            state["messages"].append(HumanMessage(content=user_input))
            state = graph.invoke(state)
            entities_touched.update(state["mentioned_entities"])

            ai_reply = state["messages"][-1].content
            print(f"agent> {ai_reply}\n")
    except EOFError:
        pass

    if len(state["messages"]) > 0:
        summary = write_episode(session_id, state["messages"], list(entities_touched))
        db = get_session()
        try:
            close_session(db, session_id, summary)
            db.commit()
        finally:
            db.close()
        print(f"\nSession closed. Summary written to episodic memory:\n{summary}")


if __name__ == "__main__":
    main()
