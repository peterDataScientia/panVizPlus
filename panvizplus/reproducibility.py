"""Reproducibility and publication-export helpers for panVizPlus."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone

import yaml

from panvizplus import __version__
from panvizplus.interactions.models import InteractionRecord


def interaction_to_dict(record: InteractionRecord) -> dict:
    return {
        "interaction_id": record.interaction_id,
        "interaction_type": record.interaction_type,
        "ligand_site": record.ligand_site,
        "protein_site": record.protein_site,
        "residue_name": record.residue_name,
        "residue_number": record.residue_number,
        "chain_id": record.chain_id,
        "detector": record.detector,
        "detector_version": record.detector_version,
        "ruleset": record.ruleset,
        "origin": record.origin,
        "measurements": dict(record.measurements),
        "criteria": [
            {
                "name": c.name,
                "measured_value": c.measured_value,
                "comparator": c.comparator,
                "threshold": c.threshold,
                "units": c.units,
                "passed": c.passed,
            }
            for c in record.criteria
        ],
        "metadata": dict(record.metadata),
    }


def build_manifest(
    *,
    input_sha256: str,
    ligand_selector: str,
    ligand_net_charge: int,
    chemistry: dict,
    rules: dict,
    interaction_count: int,
    pose_score: float | None = None,
) -> dict:
    return {
        "schema": "panVizPlus-analysis-manifest-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "panvizplus_version": __version__,
        "input_sha256": input_sha256,
        "ligand_selector": ligand_selector,
        "ligand_net_charge": int(ligand_net_charge),
        "chemistry": chemistry,
        "ruleset": rules.get("metadata", {}),
        "interaction_count": int(interaction_count),
        "pose_score_kcal_mol": pose_score,
    }


def build_methods_text(manifest: dict) -> str:
    chem = manifest.get("chemistry", {})
    rule = manifest.get("ruleset", {})
    return (
        "Protein–ligand interactions were analyzed with panVizPlus "
        f"{manifest.get('panvizplus_version')}. Ligand chemistry was resolved using "
        f"{chem.get('source', 'unspecified chemistry source')} "
        f"(mode: {chem.get('reconstruction_mode', 'unspecified')}; "
        f"reported ligand net charge: {manifest.get('ligand_net_charge'):+d}). "
        "RDKit was used for ligand chemical perception where supported, while "
        "panVizPlus applied explicit protein–ligand geometric interaction criteria. "
        f"The rule profile was {rule.get('display_name', rule.get('id', 'unspecified'))} "
        f"(version {rule.get('version', 'unspecified')}). "
        "Automatically detected interactions and their per-criterion provenance "
        "were retained for reproducibility."
    )


def build_figure_caption(manifest: dict) -> str:
    rule = manifest.get("ruleset", {})
    return (
        "Two-dimensional protein–ligand interaction map generated with panVizPlus. "
        "Residue labels are positioned from local ligand interaction anchors using "
        "collision-aware, non-circular placement; the ligand is rendered as a "
        "chemical skeletal structure. Interaction colors and line styles denote "
        "interaction classes detected under the "
        f"{rule.get('display_name', rule.get('id', 'selected'))} rule profile."
    )


def build_publication_bundle(
    *,
    svg: str,
    records: list[InteractionRecord],
    manifest: dict,
    rules: dict,
    audit: dict,
) -> bytes:
    record_dicts = [interaction_to_dict(r) for r in records]
    methods = build_methods_text(manifest)
    caption = build_figure_caption(manifest)

    csv_buf = io.StringIO()
    writer = csv.DictWriter(
        csv_buf,
        fieldnames=[
            "interaction_id", "interaction_type", "residue", "ligand_site",
            "protein_site", "ruleset", "origin", "chemistry_source",
        ],
    )
    writer.writeheader()
    for r in records:
        writer.writerow({
            "interaction_id": r.interaction_id,
            "interaction_type": r.interaction_type,
            "residue": f"{r.residue_name}{r.residue_number}:{r.chain_id or '-'}",
            "ligand_site": r.ligand_site,
            "protein_site": r.protein_site,
            "ruleset": r.ruleset,
            "origin": r.origin,
            "chemistry_source": (
                r.metadata.get("chemistry_source")
                or r.metadata.get("ligand_feature_source")
            ),
        })

    audit_export = dict(audit)
    hbond = dict(audit_export.get("hbond_audit", {}))
    hbond.pop("records", None)
    audit_export["hbond_audit"] = hbond

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("figure.svg", svg)
        zf.writestr("interactions.csv", csv_buf.getvalue())
        zf.writestr(
            "interactions.json",
            json.dumps(record_dicts, indent=2, ensure_ascii=False),
        )
        zf.writestr(
            "analysis_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )
        zf.writestr("rules_used.yaml", yaml.safe_dump(rules, sort_keys=False))
        zf.writestr(
            "audit_summary.json",
            json.dumps(audit_export, indent=2, ensure_ascii=False),
        )
        zf.writestr("methods.txt", methods)
        zf.writestr("figure_caption.txt", caption)
    return output.getvalue()
