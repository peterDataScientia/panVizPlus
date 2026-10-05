"""Conventional hydrogen-bond detection and diagnostics for panVizPlus."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

from panvizplus.chemistry.features import ChemicalFeature, perceive_hbond_features
from panvizplus.chemistry.models import Atom, NormalizedStructure
from panvizplus.interactions.geometry import angle_degrees, atom_distance
from panvizplus.interactions.models import CriterionResult, InteractionRecord
from panvizplus.rules import load_ruleset


@dataclass(frozen=True, slots=True)
class _Orientation:
    donor: ChemicalFeature
    acceptor: ChemicalFeature
    ligand_is_donor: bool


def audit_conventional_hbonds(
    structure: NormalizedStructure,
    ligand_residue_name: str | None = None,
    ruleset: dict | None = None,
) -> list[InteractionRecord]:
    """Evaluate nearby donor/acceptor candidates and retain PASS/FAIL evidence.

    This diagnostic API is intentionally broader than detection. Candidates are
    screened by a non-detection distance radius so rejected pairs remain
    inspectable. A candidate is accepted only when every active detection
    criterion passes.
    """
    ruleset = ruleset or load_ruleset()
    rule = ruleset["interactions"]["conventional_hbond"]
    geometry = rule["geometry"]
    diagnostics = rule.get("diagnostics", {})

    da_max = float(geometry["donor_acceptor_distance_max"])
    screen_max = float(diagnostics.get("candidate_distance_max", max(4.0, da_max)))
    dha_min = float(geometry.get("dha_angle_min", 90.0))
    hay_min = float(geometry.get("hay_angle_min", 90.0))
    xda_min = float(geometry.get("xda_angle_min", 90.0))
    day_min = float(geometry.get("day_angle_min", 90.0))

    protein_features, ligand_features = perceive_hbond_features(
        structure, ligand_residue_name
    )
    p_donors = [f for f in protein_features if f.kind == "hydrogen_donor"]
    p_acceptors = [f for f in protein_features if f.kind == "hydrogen_acceptor"]
    l_donors = [f for f in ligand_features if f.kind == "hydrogen_donor"]
    l_acceptors = [f for f in ligand_features if f.kind == "hydrogen_acceptor"]

    orientations = [
        *(_Orientation(d, a, False) for d in p_donors for a in l_acceptors),
        *(_Orientation(d, a, True) for d in l_donors for a in p_acceptors),
    ]

    amap = structure.atom_map()
    results: list[InteractionRecord] = []
    serial = count(1)

    for pair in orientations:
        donor = amap[pair.donor.atom_id]
        acceptor = amap[pair.acceptor.atom_id]
        da = atom_distance(donor, acceptor)
        if da > screen_max:
            continue

        criteria = [
            CriterionResult(
                "donor_acceptor_distance",
                round(da, 4),
                "<=",
                da_max,
                "angstrom",
                da <= da_max,
            )
        ]
        measurements: dict[str, float | str | bool | None] = {
            "donor_acceptor_distance": round(da, 4)
        }
        geometry_mode = "heavy_atom_surrogate"

        explicit_h = _best_explicit_hydrogen(pair.donor, acceptor, amap)
        if explicit_h is not None:
            geometry_mode = "explicit_hydrogen"
            dha = angle_degrees(donor, explicit_h, acceptor)
            measurements["DHA_angle"] = None if dha is None else round(dha, 3)
            criteria.append(
                CriterionResult(
                    "DHA_angle",
                    measurements["DHA_angle"],
                    ">=",
                    dha_min,
                    "degree",
                    dha is not None and dha >= dha_min,
                )
            )
            hay = _best_acceptor_angle(explicit_h, acceptor, pair.acceptor, amap)
            if hay is not None:
                measurements["HAY_angle"] = round(hay, 3)
                criteria.append(
                    CriterionResult(
                        "HAY_angle",
                        round(hay, 3),
                        ">=",
                        hay_min,
                        "degree",
                        hay >= hay_min,
                    )
                )
        else:
            xda = _best_donor_surrogate_angle(pair.donor, donor, acceptor, amap)
            if xda is not None:
                measurements["XDA_angle"] = round(xda, 3)
                criteria.append(
                    CriterionResult(
                        "XDA_angle",
                        round(xda, 3),
                        ">=",
                        xda_min,
                        "degree",
                        xda >= xda_min,
                    )
                )
            day = _best_acceptor_angle(donor, acceptor, pair.acceptor, amap)
            if day is not None:
                measurements["DAY_angle"] = round(day, 3)
                criteria.append(
                    CriterionResult(
                        "DAY_angle",
                        round(day, 3),
                        ">=",
                        day_min,
                        "degree",
                        day >= day_min,
                    )
                )

        angle_criteria = [c for c in criteria if c.name.endswith("_angle")]
        if not angle_criteria:
            criteria.append(
                CriterionResult(
                    "directional_geometry_available",
                    False,
                    "==",
                    True,
                    None,
                    False,
                )
            )

        accepted = all(c.passed for c in criteria)
        rejection_reasons = [c.name for c in criteria if not c.passed]

        ligand_atom = donor if pair.ligand_is_donor else acceptor
        protein_atom = acceptor if pair.ligand_is_donor else donor
        ligand_feature = pair.donor if pair.ligand_is_donor else pair.acceptor
        protein_feature = pair.acceptor if pair.ligand_is_donor else pair.donor

        results.append(
            InteractionRecord(
                interaction_id=f"HBOND-AUDIT-{next(serial):04d}",
                interaction_type="conventional_hbond",
                ligand_site=_site_label(ligand_atom),
                protein_site=_site_label(protein_atom),
                residue_name=protein_atom.residue_name,
                residue_number=protein_atom.residue_number,
                chain_id=protein_atom.chain_id or None,
                detector="panVizPlus-native",
                detector_version=str(ruleset["metadata"]["version"]),
                ruleset=ruleset["metadata"]["id"],
                criteria=criteria,
                measurements=measurements,
                metadata={
                    "audit_status": "accepted" if accepted else "rejected",
                    "rejection_reasons": rejection_reasons,
                    "screening_distance_max": screen_max,
                    "geometry_mode": geometry_mode,
                    "donor_site": _site_label(donor),
                    "acceptor_site": _site_label(acceptor),
                    "ligand_role": "donor" if pair.ligand_is_donor else "acceptor",
                    "ligand_feature_confidence": ligand_feature.confidence,
                    "ligand_feature_source": ligand_feature.source,
                    "protein_feature_confidence": protein_feature.confidence,
                    "protein_feature_source": protein_feature.source,
                    "rule_status": rule.get("status"),
                    "ruleset_exact_biovia_reproduction": ruleset["metadata"].get(
                        "exact_biovia_reproduction", False
                    ),
                },
            )
        )

    return _deduplicate(results)


def detect_conventional_hbonds(
    structure: NormalizedStructure,
    ligand_residue_name: str | None = None,
    ruleset: dict | None = None,
) -> list[InteractionRecord]:
    """Return only candidates that pass every active H-bond criterion."""
    return [
        record
        for record in audit_conventional_hbonds(
            structure,
            ligand_residue_name=ligand_residue_name,
            ruleset=ruleset,
        )
        if record.metadata.get("audit_status") == "accepted"
    ]


def _best_explicit_hydrogen(
    donor_feature: ChemicalFeature,
    acceptor: Atom,
    amap: dict[int, Atom],
) -> Atom | None:
    candidates = [amap[i] for i in donor_feature.hydrogen_ids if i in amap]
    return min(candidates, key=lambda h: atom_distance(h, acceptor)) if candidates else None


def _best_donor_surrogate_angle(
    donor_feature: ChemicalFeature,
    donor: Atom,
    acceptor: Atom,
    amap: dict[int, Atom],
) -> float | None:
    values = [
        angle_degrees(amap[nid], donor, acceptor)
        for nid in donor_feature.neighbor_ids
        if nid in amap
    ]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _best_acceptor_angle(
    first: Atom,
    acceptor: Atom,
    acceptor_feature: ChemicalFeature,
    amap: dict[int, Atom],
) -> float | None:
    values = [
        angle_degrees(first, acceptor, amap[nid])
        for nid in acceptor_feature.neighbor_ids
        if nid in amap
    ]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _site_label(atom: Atom) -> str:
    return f"{atom.residue_name}:{atom.chain_id or '-'}:{atom.residue_number}:{atom.name}"


def _deduplicate(records: list[InteractionRecord]) -> list[InteractionRecord]:
    """Keep one physical ligand-protein atom-pair H-bond.

    A donor/acceptor pair can be perceived in both directions when both sites
    are chemically amphoteric. The displayed scientific interaction is the
    atom pair, so retain the geometrically better orientation instead of
    reporting two copies of the same contact.
    """
    best: dict[tuple[str, str, str], InteractionRecord] = {}
    for record in records:
        key = (record.ligand_site, record.protein_site, record.interaction_type)
        current = best.get(key)
        if current is None or _geometry_score(record) > _geometry_score(current):
            best[key] = record
    return list(best.values())


def _geometry_score(record: InteractionRecord) -> tuple[int, float]:
    explicit = 1 if record.metadata.get("geometry_mode") == "explicit_hydrogen" else 0
    angles = [
        float(value)
        for name, value in record.measurements.items()
        if name.endswith("_angle") and isinstance(value, (int, float))
    ]
    return explicit, min(angles) if angles else 0.0
