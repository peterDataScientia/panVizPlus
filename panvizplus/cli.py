from __future__ import annotations

import argparse

from panvizplus.chemistry.pdb import read_pdb
from panvizplus.interactions.engine import analyze_structure


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="panvizplus",
        description="Run the native panVizPlus protein-ligand interaction engine.",
    )
    parser.add_argument("structure", help="PDB structure file")
    parser.add_argument("--ligand", required=True, help="Ligand selector RESNAME:CHAIN:RESNUM")
    args = parser.parse_args()

    structure = read_pdb(args.structure)
    records = analyze_structure(structure, args.ligand)
    for warning in structure.warnings:
        print(f"WARNING: {warning}")

    if not records:
        print("No interactions passed the current panVizPlus rules.")
        return 0

    print("ID\tType\tLigand\tProtein\tResidue\tRuleset")
    for r in records:
        print(
            f"{r.interaction_id}\t{r.interaction_type}\t{r.ligand_site}\t"
            f"{r.protein_site}\t{r.residue_name}{r.residue_number}:{r.chain_id or '-'}\t{r.ruleset}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
