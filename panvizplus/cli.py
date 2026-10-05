from __future__ import annotations

import argparse

from panvizplus.analysis import analyze_pdb


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="panvizplus",
        description="Audit protein-ligand interactions with panVizPlus.",
    )
    parser.add_argument("structure", help="PDB structure file")
    parser.add_argument("--ligand", help="Ligand residue name")
    args = parser.parse_args()

    interactions, warnings = analyze_pdb(args.structure, args.ligand)
    for warning in warnings:
        print(f"WARNING: {warning}")

    if not interactions:
        print("No confirmed conventional protein-ligand H-bonds found.")
        return 0

    print("ID\tLigand\tProtein\tD-A(A)\tGeometry\tLigand perception\tRule set")
    for item in interactions:
        print(
            f"{item.interaction_id}\t"
            f"{item.ligand_site}\t"
            f"{item.protein_site}\t"
            f"{item.measurements.get('donor_acceptor_distance')}\t"
            f"{item.metadata.get('geometry_mode')}\t"
            f"{item.metadata.get('ligand_feature_confidence')}\t"
            f"{item.ruleset}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
