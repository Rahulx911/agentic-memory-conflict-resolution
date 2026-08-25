"""The four metrics tracked in the README: memory retrieval precision,
conflict-resolution accuracy, cross-session recall, and latency.

Conflict-resolution accuracy is pure (scores `src.conflict.policy.decide`
against a labeled set, no I/O). Retrieval precision and cross-session recall
need a DB session, since they exercise `src.agent.nodes.perceive` and
`src.memory.repository` against real entities/facts. Latency just does
percentile math over caller-supplied timing samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session as DBSession

from src.agent.nodes import perceive
from src.conflict import policy
from src.eval.contradiction_set import CONTRADICTION_SET, LabeledContradiction
from src.eval.mention_set import MENTION_SET, MentionCase
from src.memory import repository

# ---------------------------------------------------------------------------
# Conflict-resolution accuracy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConflictCaseResult:
    case_id: str
    correct: bool
    expected_auto_resolved: bool
    actual_auto_resolved: bool
    expected_new_is_current: bool | None
    actual_new_is_current: bool | None


@dataclass(frozen=True)
class ConflictAccuracyReport:
    results: list[ConflictCaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def mismatches(self) -> list[ConflictCaseResult]:
        return [r for r in self.results if not r.correct]


def conflict_resolution_accuracy(
    cases: list[LabeledContradiction] = CONTRADICTION_SET,
) -> ConflictAccuracyReport:
    results = []
    for case in cases:
        decision = policy.decide(
            entity_type=case.entity_type,
            attribute=case.attribute,
            candidate_confidence=case.candidate_confidence,
            candidate_observed_at=case.candidate_observed_at,
            current_observed_at=case.current_observed_at,
        )
        auto_matches = decision.auto_resolved == case.expected_auto_resolved
        # new_is_current is only a meaningful label when auto-resolution was
        # expected at all; a flagged-for-human case has no "correct side."
        current_matches = (
            decision.new_is_current == case.expected_new_is_current
            if case.expected_auto_resolved
            else True
        )
        results.append(
            ConflictCaseResult(
                case_id=case.case_id,
                correct=auto_matches and current_matches,
                expected_auto_resolved=case.expected_auto_resolved,
                actual_auto_resolved=decision.auto_resolved,
                expected_new_is_current=case.expected_new_is_current,
                actual_new_is_current=decision.new_is_current if case.expected_auto_resolved else None,
            )
        )
    return ConflictAccuracyReport(results=results)


# ---------------------------------------------------------------------------
# Memory retrieval precision (entity-mention detection feeding retrieve_memory)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MentionCaseResult:
    case_id: str
    text: str
    expected: frozenset[str]
    actual: frozenset[str]

    @property
    def false_positives(self) -> frozenset[str]:
        return self.actual - self.expected

    @property
    def false_negatives(self) -> frozenset[str]:
        return self.expected - self.actual

    @property
    def correct(self) -> bool:
        return self.actual == self.expected


@dataclass(frozen=True)
class RetrievalPrecisionReport:
    results: list[MentionCaseResult]

    @property
    def precision(self) -> float:
        """Of everything retrieved as 'mentioned' across all cases, what fraction
        was actually mentioned. 1.0 means no false positives."""
        retrieved = sum(len(r.actual) for r in self.results)
        true_positives = sum(len(r.actual & r.expected) for r in self.results)
        return true_positives / retrieved if retrieved else 1.0

    @property
    def recall(self) -> float:
        """Of everything that should have been retrieved, what fraction was."""
        expected = sum(len(r.expected) for r in self.results)
        true_positives = sum(len(r.actual & r.expected) for r in self.results)
        return true_positives / expected if expected else 1.0

    @property
    def exact_match_rate(self) -> float:
        return sum(1 for r in self.results if r.correct) / len(self.results) if self.results else 1.0


def memory_retrieval_precision(cases: list[MentionCase] = MENTION_SET) -> RetrievalPrecisionReport:
    """Requires the entities in `src.eval.mention_set.FIXTURE_ENTITIES` to already
    exist in the DB perceive() reads from (get_session() opens its own connection,
    so this only needs those rows committed beforehand, not a passed-in session)."""
    results = []
    for case in cases:
        state = {"messages": [HumanMessage(content=case.text)]}
        mentioned = perceive(state)["mentioned_entities"]
        actual = frozenset(e["name"] for e in mentioned)
        results.append(
            MentionCaseResult(case_id=case.case_id, text=case.text, expected=case.expected_entities, actual=actual)
        )
    return RetrievalPrecisionReport(results=results)


# ---------------------------------------------------------------------------
# Cross-session recall (structured facts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecallExpectation:
    entity_type: str
    entity_name: str
    attribute: str
    expected_value: dict


def structured_cross_session_recall(db: DBSession, expectations: list[RecallExpectation]) -> float:
    """Fraction of `expectations` still correctly current — meant to be called
    after simulating several intervening sessions since the facts were written,
    to demonstrate (and regression-guard) that structured memory isn't session-scoped."""
    if not expectations:
        return 1.0
    correct = 0
    for exp in expectations:
        entity = repository.find_entity_by_name(db, exp.entity_name)
        if entity is None:
            continue
        current = repository.get_current_fact(db, entity.id, exp.attribute)
        if current is not None and current.value == exp.expected_value:
            correct += 1
    return correct / len(expectations)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyReport:
    samples_seconds: list[float] = field(default_factory=list)

    def _percentile(self, p: float) -> float:
        if not self.samples_seconds:
            return 0.0
        ordered = sorted(self.samples_seconds)
        idx = min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1)
        return ordered[max(idx, 0)]

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def mean(self) -> float:
        return sum(self.samples_seconds) / len(self.samples_seconds) if self.samples_seconds else 0.0
