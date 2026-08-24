# Architecture Notes

The original spec included an architecture diagram that wasn't captured in text form. Recording the described flow here until a diagram is added.

## Memory tiers

- **Structured memory (Postgres):** entities (equipment, zones) → facts (status, last-serviced, ...) each with `confidence`, `timestamp`, `source_session_id`, and a `provenance` audit trail of prior values.
- **Episodic/semantic memory (Qdrant):** per-session conversation summaries embedded for semantic recall, keyed to session metadata (date, participants, entities touched).

## Agent loop (LangGraph)

```
perceive -> retrieve relevant memory -> reason/act (tools) -> respond -> write memory updates
```

- **retrieve relevant memory:** structured lookups for entities named in the turn + Qdrant similarity search over episodic summaries.
- **reason/act:** tool calls — DB lookups, incident search, human escalation.
- **write memory updates:** memory-extraction node proposes candidate facts; conflict-resolution node checks against existing structured facts before commit.

## Conflict resolution policy

1. New candidate fact extracted from a turn.
2. Check structured memory for an existing fact on the same (entity, attribute).
3. If no conflict → write directly, log provenance.
4. If conflict:
   - Low-stakes (heuristic: attribute not in high-stakes list, or confidence delta small) → auto-resolve by recency, log both old and new value + reason.
   - High-stakes (safety-relevant equipment status, etc.) → do not auto-overwrite; surface to user for confirmation, hold prior fact as current until confirmed.
5. Every write/overwrite logged with session id, source, and resolution reasoning for auditability.
