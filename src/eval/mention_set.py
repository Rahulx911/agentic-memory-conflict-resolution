"""Labeled entity-mention set for the memory-retrieval-precision metric.

`retrieve_memory` only fetches facts for entities `perceive` decided were
"mentioned" in the user's message — and `perceive` decides that with a plain
substring check (`entity.name.lower() in text`). That's cheap but has an
obvious failure mode: a short entity name that's a substring of a longer one
("bay_1" inside "bay_12") gets flagged as mentioned even when it wasn't. This
set is built specifically to surface that, rather than only covering the easy
cases.
"""

from __future__ import annotations

from dataclasses import dataclass

# (entity_type, name) pairs the mention cases below assume exist in the DB —
# the eval harness / tests are responsible for seeding these before scoring.
FIXTURE_ENTITIES: list[tuple[str, str]] = [
    ("equipment", "sensor_3"),
    ("equipment", "sensor_30"),
    ("zone", "bay_1"),
    ("zone", "bay_12"),
    ("zone", "loading_dock"),
]


@dataclass(frozen=True)
class MentionCase:
    case_id: str
    text: str
    expected_entities: frozenset[str]


MENTION_SET: list[MentionCase] = [
    MentionCase(
        case_id="single_unambiguous_mention",
        text="What's the status on sensor_3 right now?",
        expected_entities=frozenset({"sensor_3"}),
    ),
    MentionCase(
        case_id="two_distinct_mentions",
        text="Has sensor_3 or bay_1 had any incidents this week?",
        expected_entities=frozenset({"sensor_3", "bay_1"}),
    ),
    MentionCase(
        case_id="no_entities_mentioned",
        text="How is the team doing today?",
        expected_entities=frozenset(),
    ),
    MentionCase(
        case_id="short_name_is_substring_of_longer_name",
        text="The forklift just moved into bay_12.",
        expected_entities=frozenset({"bay_12"}),
    ),
    MentionCase(
        case_id="short_equipment_name_is_substring_of_longer_name",
        text="sensor_30 just came back online after maintenance.",
        expected_entities=frozenset({"sensor_30"}),
    ),
    MentionCase(
        case_id="both_short_and_long_name_present",
        text="sensor_3 is fine, but sensor_30 needs a look.",
        expected_entities=frozenset({"sensor_3", "sensor_30"}),
    ),
]
