from __future__ import annotations

from pathlib import Path

from panvizplus.chemistry.pdb import read_pdb
from panvizplus.interactions.hydrogen_bond import detect_conventional_hbonds
from panvizplus.interactions.models import InteractionRecord


def analyze_pdb(
    path: str | Path,
    ligand_residue_name: str | None = None,
) -> tuple[list[InteractionRecord], list[str]]:
    structure = read_pdb(path)
    interactions = detect_conventional_hbonds(
        structure,
        ligand_residue_name=ligand_residue_name,
    )
    return interactions, structure.warnings
