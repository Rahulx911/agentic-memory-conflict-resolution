"""Human-operator review queue for conflicts the policy couldn't auto-resolve.

This is the other end of src.conflict.policy: every high-stakes conflict is
staged as PENDING_CONFIRMATION and logged as an open Escalation. This CLI
lists them and lets an operator accept the new observation or reject it and
keep the current one.

    python -m src.conflict.review_cli
"""

from dotenv import load_dotenv

load_dotenv()

from src.memory import repository
from src.memory.db import get_session, init_db
from src.memory.models import Entity, Fact


def _describe(db, fact: Fact) -> str:
    entity = db.get(Entity, fact.entity_id)
    label = f"{entity.entity_type}:{entity.name}" if entity else str(fact.entity_id)
    lines = [f"[{fact.id}] {label}.{fact.attribute}"]
    lines.append(f"    new observation:  {fact.value} (confidence {fact.confidence}, observed {fact.observed_at})")
    if fact.supersedes_fact_id:
        prior = db.get(Fact, fact.supersedes_fact_id)
        if prior is not None:
            lines.append(f"    current fact:     {prior.value} (confidence {prior.confidence}, observed {prior.observed_at})")
    lines.append(f"    why flagged:      {fact.resolution_reason}")
    return "\n".join(lines)


def main() -> None:
    init_db()
    db = get_session()
    try:
        pending = repository.list_pending_conflicts(db)
        if not pending:
            print("No pending conflicts.")
            return

        print(f"{len(pending)} pending conflict(s):\n")
        for fact in pending:
            print(_describe(db, fact))
            choice = ""
            while choice not in {"a", "r", "s"}:
                choice = input("  [a]ccept new / [r]eject (keep current) / [s]kip: ").strip().lower()
            if choice == "s":
                print()
                continue
            repository.confirm_conflict(db, fact.id, accept=(choice == "a"))
            db.commit()
            print("  -> recorded.\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
