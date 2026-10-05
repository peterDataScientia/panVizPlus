"""Scientific audit summaries for panVizPlus analyses."""

from __future__ import annotations

from collections import Counter

from panvizplus.chemistry.models import NormalizedStructure, distance
from panvizplus.chemistry.rdkit_layer import LigandChemistry
from panvizplus.interactions.hydrogen_bond import audit_conventional_hbonds
from panvizplus.interactions.models import InteractionRecord


def build_analysis_audit(
    structure: NormalizedStructure,
    ligand_selector: str,
    records: list[InteractionRecord],
    ruleset: dict,
    chemistry: LigandChemistry,
    screening_radius: float = 6.5,
) -> dict:
    ligand = [a for a in structure.ligand_atoms(ligand_selector) if not a.is_hydrogen]
    protein = [a for a in structure.protein_atoms() if not a.is_hydrogen]

    nearby_atoms = []
    nearby_residues = set()
    if ligand:
        for pa in protein:
            if any(distance(pa, la) <= screening_radius for la in ligand):
                nearby_atoms.append(pa)
                nearby_residues.add((pa.chain_id, pa.residue_number, pa.residue_name))

    hbond_candidates = audit_conventional_hbonds(
        structure,
        ligand_selector,
        ruleset,
        ligand_chemistry=chemistry,
    )

    hbond_reasons = Counter()
    for rec in hbond_candidates:
        if rec.metadata.get("audit_status") != "accepted":
            for reason in rec.metadata.get("rejection_reasons", []):
                hbond_reasons[str(reason)] += 1

    accepted_by_type = Counter(r.interaction_type for r in records)
    fallback_sources = Counter()
    for rec in records:
        source = (
            rec.metadata.get("chemistry_source")
            or rec.metadata.get("ligand_feature_source")
        )
        if source and "fallback" in str(source).lower():
            fallback_sources[str(source)] += 1

    return {
        "screening_radius": float(screening_radius),
        "ligand_heavy_atoms": len(ligand),
        "protein_heavy_atoms_near_ligand": len(nearby_atoms),
        "protein_residues_near_ligand": len(nearby_residues),
        "chemistry": {
            "source": chemistry.source,
            "reconstruction_mode": chemistry.reconstruction_mode,
            "confidence": chemistry.confidence,
            "donors": len(chemistry.donor_atom_ids),
            "acceptors": len(chemistry.acceptor_atom_ids),
            "positive_sites": len(chemistry.positive_atom_ids),
            "negative_sites": len(chemistry.negative_atom_ids),
            "hydrophobes": len(chemistry.hydrophobe_atom_ids),
            "aromatic_rings": len(chemistry.aromatic_rings),
        },
        "accepted_by_type": dict(sorted(accepted_by_type.items())),
        "accepted_total": len(records),
        "hbond_audit": {
            "candidate_total": len(hbond_candidates),
            "accepted": sum(
                1 for r in hbond_candidates
                if r.metadata.get("audit_status") == "accepted"
            ),
            "rejected": sum(
                1 for r in hbond_candidates
                if r.metadata.get("audit_status") != "accepted"
            ),
            "rejection_reasons": dict(sorted(hbond_reasons.items())),
            "records": hbond_candidates,
            "scope": "full_pass_fail",
        },
        "fallback_sources": dict(sorted(fallback_sources.items())),
        "audit_scope": {
            "conventional_hbond": "full_pass_fail",
            "other_interaction_classes": "accepted_only_plus_spatial_summary",
        },
    }


def hbond_audit_rows(audit: dict) -> list[dict]:
    rows = []
    for rec in audit.get("hbond_audit", {}).get("records", []):
        rows.append({
            "Status": rec.metadata.get("audit_status", "unknown"),
            "Residue": f"{rec.residue_name}{rec.residue_number}:{rec.chain_id or '-'}",
            "Ligand site": rec.ligand_site,
            "Protein site": rec.protein_site,
            "Distance (Å)": rec.measurements.get("donor_acceptor_distance"),
            "Geometry mode": rec.metadata.get("geometry_mode"),
            "Rejection reason": ", ".join(rec.metadata.get("rejection_reasons", [])),
            "Criteria": "; ".join(
                f"{c.name}={c.measured_value} {c.comparator} {c.threshold}"
                + (f" {c.units}" if c.units else "")
                + (" PASS" if c.passed else " FAIL")
                for c in rec.criteria
            ),
            "Chemistry source": (
                rec.metadata.get("ligand_feature_source")
                or rec.metadata.get("chemistry_source")
            ),
        })
    return rows
