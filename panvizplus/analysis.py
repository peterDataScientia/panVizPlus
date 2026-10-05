from __future__ import annotations

from pathlib import Path

from panvizplus.chemistry.pdb import read_pdb
from panvizplus.interactions.engine import analyze_structure
from panvizplus.interactions.models import InteractionRecord
from panvizplus.rules import load_ruleset


def analyze_pdb(
    path: str | Path,
    ligand_selector: str,
    ruleset: dict | None = None,
) -> tuple[list[InteractionRecord], list[str]]:
    """Analyze one ligand residue with the native panVizPlus engine."""
    structure = read_pdb(path)
    rules = ruleset or load_ruleset()
    interactions = analyze_structure(structure, ligand_selector, rules)
    return interactions, structure.warnings
