"""Guided-extraction hints that bias graphiti's node extraction.

Graphiti drops an edge when an endpoint *name* isn't in the extracted node set
(edge_operations.py: "Target entity not found in nodes for edge relation"). On terse,
value-centric text the node step often never creates an entity for a value/role/system,
so the edge has nowhere to attach. These entity-type hints + instructions bias the node
step to capture those as entities, which lets the edges resolve. Measured effect:
window-recall 1/5 -> 3/5 on the probe set (see results/cross_episode/).

Pure / no graph dependency, so this is importable and testable on its own.
"""

from __future__ import annotations

from pydantic import BaseModel


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


GUIDED_ENTITY_TYPES: dict[str, type] = {"Value": Value, "Role": Role, "System": System}

GUIDED_EXTRACTION_INSTRUCTIONS = (
    "Also extract concrete values, quantities, dates, roles, and named teams / services / "
    "products / tools as their own entities (e.g. '40 requests per minute', 'March 14', "
    "'security lead', 'billing service', 'SQLite', 'Northwind'), so that relations BETWEEN "
    "them can be formed. Prefer creating an entity for the object of a statement over omitting it."
)
