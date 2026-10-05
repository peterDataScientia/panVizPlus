from __future__ import annotations

import argparse

from panvizplus.chemistry.pdb import read_pdb
from panvizplus.interactions.engine import analyze_structure


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="panvizplus",
        description="Run RDKit chemistry perception and panVizPlus interaction rules.",
    )
    parser.add_argument("structure", help="PDB structure file")
    parser.add_argument("--ligand", required=True, help="Ligand selector RESNAME:CHAIN:RESNUM")
    parser.add_argument(
        "--ligand-charge",
        type=int,
        default=0,
        help="Ligand net formal charge used for RDKit bond-order perception.",
    )
    args = parser.parse_args()

    structure = read_pdb(args.structure)
    records = analyze_structure(
        structure,
        args.ligand,
        ligand_net_charge=args.ligand_charge,
    )
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
