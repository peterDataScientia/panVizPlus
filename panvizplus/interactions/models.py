"""Core provenance records for panVizPlus interactions.

The renderer should consume these records; it should never infer chemistry itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """One measured geometric/chemical criterion used by a detector."""

    name: str
    measured_value: float | str | bool | None
    comparator: str
    threshold: float | str | bool | None
    units: str | None
    passed: bool


@dataclass(slots=True)
class InteractionRecord:
    """Auditable representation of one detected or manually curated interaction."""

    interaction_id: str
    interaction_type: str
    ligand_site: str
    protein_site: str
    residue_name: str
    residue_number: int
    chain_id: str | None = None
    detector: str = "panVizPlus"
    detector_version: str = "0.1.0.dev0"
    ruleset: str = "panvizplus_v1"
    origin: str = "automatic"
    criteria: list[CriterionResult] = field(default_factory=list)
    measurements: dict[str, float | str | bool | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passes_all_criteria(self) -> bool:
        return bool(self.criteria) and all(item.passed for item in self.criteria)
