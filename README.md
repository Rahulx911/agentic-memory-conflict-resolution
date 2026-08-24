# Agentic System with Long-Term Memory & Conflict Resolution

**Target audience:** Reusable agentic-AI infrastructure — applicable to both banking/compliance-focused evaluation (audit trail, explainability of memory writes) and product-focused evaluation (retrieval precision, conflict-resolution accuracy as measurable metrics).

## Problem Statement

Most agent demos are stateless — every session starts from zero. This project builds an agent (reference implementation: a warehouse-operations assistant) that maintains **persistent memory** — facts about equipment, past incidents, and user preferences — across sessions, updates its beliefs as new information arrives, and explicitly resolves contradictions instead of silently overwriting or ignoring them (e.g. "the fault log said sensor 3 was replaced, but a new log says it's still faulty — which is current?").

The differentiating engineering problem is not memory storage, it's **conflict resolution with provenance**: every fact carries a confidence and timestamp, every overwrite is logged with its source session, and high-stakes contradictions are surfaced to a human rather than auto-resolved.

## Architecture

Two-tier long-term memory:

- **Structured memory (Postgres)** — entities (equipment, zones), facts (status, last-serviced, confidence, timestamp), and a full write/overwrite audit log for provenance.
- **Episodic / semantic memory (Qdrant)** — conversation summaries with embeddings for semantic recall across sessions.

Agent loop (LangGraph state machine): `perceive → retrieve relevant memory → reason/act (tool calls: DB lookup, incident search, human escalation) → respond → extract & write memory updates`.

Before any new fact is written, it is checked against existing structured memory for conflicts. Low-stakes conflicts auto-resolve by recency; high-stakes conflicts are flagged for user confirmation. Every write is logged with session and source, for auditability.

## Build Plan

- [ ] **Phase 1 — Memory Schema (Week 1)**
  - [ ] Structured memory schema: entities, facts, confidence/timestamp per fact
  - [ ] Episodic memory schema: conversation summaries + embeddings
  - [ ] Postgres (structured facts) + Qdrant (episodic/semantic) set up
- [x] **Phase 2 — Agent Core (Week 2)**
  - [x] LangGraph state machine: perceive → retrieve → reason/act → respond → write
  - [x] Tool calls: DB lookups, incident search, human escalation
  - [x] Memory-extraction node (LLM extracts candidate new facts each turn)
- [ ] **Phase 3 — Conflict Resolution (Week 3, the differentiating part)**
  - [ ] Conflict check against existing structured memory before write
  - [ ] Resolution policy: auto-resolve low-stakes by recency, flag high-stakes for confirmation
  - [ ] Provenance logging on every memory write/overwrite (session, source)
  - [ ] Test suite of deliberately contradictory input sequences across sessions
- [ ] **Phase 4 — Evaluation & Polish (Week 4)**
  - [ ] Multi-session recall test scenarios (session 1 fact recalled correctly in session 5)
  - [ ] Metrics: memory precision (retrieval correctness), conflict-resolution accuracy vs. labeled contradiction set, latency
  - [ ] Simple chat UI showing memory state transparently (what the agent remembers, and why)

## Metrics Tracked

| Metric | Definition | Target |
|---|---|---|
| Memory retrieval precision | % of retrieved facts relevant to query | TBD after Phase 4 baseline |
| Conflict-resolution accuracy | % correct resolutions vs. labeled contradiction set | TBD after Phase 4 baseline |
| Cross-session recall | % of session-1 facts correctly recalled by session 5+ | TBD after Phase 4 baseline |
| p50/p95 response latency | end-to-end perceive→respond | TBD |

## Tech Stack

- **Orchestration:** LangGraph
- **Structured memory:** Postgres
- **Episodic/semantic memory:** Qdrant
- **LLM:** Claude (see `docs/`)
- **UI:** minimal chat interface (memory-state transparency panel)

## Repo Layout

```
src/
  memory/    structured + episodic memory schema and access layer
  agent/     LangGraph state machine and nodes
  conflict/  conflict detection + resolution policy
  tools/     DB lookups, incident search, escalation tools
  eval/      metrics + labeled contradiction test harness
  ui/        chat UI with memory-state transparency
tests/
  fixtures/    sample entities/facts for tests
  scenarios/   multi-session, contradiction test scenarios
docker/        Postgres + Qdrant local dev stack
docs/          architecture notes
```

## Running it

```
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
docker compose -f docker/docker-compose.yml up -d
cp .env.example .env   # fill in ANTHROPIC_API_KEY and VOYAGE_API_KEY
.venv/bin/python -m src.agent.run_cli
```

## Status

Phase 1 (memory schema) and Phase 2 (agent core) done. Phase 3 (conflict resolution) is next.
