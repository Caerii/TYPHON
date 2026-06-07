"""Guided-extraction hints that bias graphiti's node extraction.

Two complementary jobs, both expressed as ``entity_types`` + instructions passed to
``add_episode``:

1. **Edge formation** (``Value`` / ``Role`` / ``System``). Graphiti drops an edge when an
   endpoint *name* isn't in the extracted node set (edge_operations.py: "Target entity not
   found in nodes for edge relation"). On terse, value-centric text the node step often
   never creates an entity for a value/role/system, so the edge has nowhere to attach.
   These hints bias the node step to capture those as entities, which lets the edges
   resolve. Measured effect: window-recall 1/5 -> 3/5 on the probe set.

2. **Current-value capture** (``Configurable``). A *changeable* thing — a rate limit, the
   primary database, a service owner — is captured as one entity whose latest value lives
   in the typed ``current_value`` attribute. Graphiti re-extracts node attributes per
   episode with the prior attributes as context and merges them, so ``current_value`` is
   **updated to the current value on supersession** (validated end-to-end: "rate limit"
   100 -> 40). This is the structured, bi-temporally-clean alternative to the LLM prose
   ``summary``, which aggregates history and therefore leaks superseded values.

``Configurable`` is the only type carrying a field, so the (extra-cost) attribute-extraction
pass runs *only* for changeable things — ``Value`` / ``Role`` / ``System`` stay field-less
and contribute classification + edge endpoints only. All four are pure pydantic models with
no graph dependency, so this module is importable and unit-testable on its own.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Value(BaseModel):
    """A concrete value, quantity, setting, limit, date, or amount stated as a fact
    (e.g. '40 requests per minute', 'March 14', '$5,000'). Use the full phrase as the
    entity name so a relation can point at it."""


class Role(BaseModel):
    """A role, title, or responsibility (e.g. 'security lead', 'billing service owner',
    'on-call engineer')."""


class System(BaseModel):
    """A team, company, service, product, system, or named tool (e.g. 'platform team',
    'billing service', 'SQLite', 'Northwind')."""


class Configurable(BaseModel):
    """A setting, limit, quantity, owner, or choice whose value can CHANGE over time
    (e.g. a rate limit, the primary database, a service owner, a deadline). Name the
    entity by its stable noun phrase (e.g. 'public API rate limit', not the value) and
    record the latest stated value in ``current_value``."""

    current_value: str | None = Field(
        None,
        description=(
            "The current value as of the latest mention, copied verbatim from the text "
            "(e.g. '40 requests per minute', 'SQLite', 'Sam', 'March 14'). When a later "
            "message states a NEW value, REPLACE the old one with the new value."
        ),
    )


GUIDED_ENTITY_TYPES: dict[str, type] = {
    "Value": Value,
    "Role": Role,
    "System": System,
    "Configurable": Configurable,
}

GUIDED_EXTRACTION_INSTRUCTIONS = (
    "Also extract concrete values, quantities, dates, roles, and named teams / services / "
    "products / tools as their own entities (e.g. '40 requests per minute', 'March 14', "
    "'security lead', 'billing service', 'SQLite', 'Northwind'), so that relations BETWEEN "
    "them can be formed. Prefer creating an entity for the object of a statement over omitting it. "
    "If a statement SETS or CHANGES a setting, limit, quantity, owner, or choice, also extract "
    "the thing being configured as a Configurable entity, named by its stable noun phrase "
    "(e.g. 'public API rate limit', 'primary database', 'billing service owner'), and record "
    "its latest value in current_value (replacing any earlier value on a change). "
    "When a concrete option is chosen FOR a purpose (e.g. 'use Postgres for the new service', "
    "'Sam now owns the billing service'), the Configurable entity is the PURPOSE / SLOT "
    "('new service database', 'billing service owner') and current_value is the chosen option "
    "('Postgres', 'Sam'); if a later message switches it ('use SQLite instead', 'reversed "
    "course'), UPDATE current_value to the new option ('SQLite'). Always reuse the same slot "
    "name so the value supersedes rather than creating a second entity."
)
