"""Native panVizPlus protein-ligand interaction engine.

This module contains no calls to third-party interaction detectors. Chemistry
perception is deliberately explicit and provenance-bearing; uncertain PDB-only
inferences are marked as such rather than silently treated as authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from math import acos, degrees, sqrt
from typing import Iterable

import numpy as np

from panvizplus.chemistry.features import ChemicalFeature, perceive_hbond_features
from panvizplus.chemistry.models import Atom, COMMON_ION_ELEMENTS, NormalizedStructure
from panvizplus.interactions.hydrogen_bond import detect_conventional_hbonds
from panvizplus.interactions.models import CriterionResult, InteractionRecord
from panvizplus.rules import load_ruleset


PROTEIN_POSITIVE = {
    ("LYS", "NZ"),
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
}
PROTEIN_NEGATIVE = {
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
}
PROTEIN_AROMATIC_RINGS = {
    "PHE": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TYR": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "HIS": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "TRP": (
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"),
    ),
}
PROTEIN_HYDROPHOBIC_ATOMS = {
    "ALA": {"CB"},
    "VAL": {"CB", "CG1", "CG2"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "MET": {"CB", "CG", "SD", "CE"},
    "PRO": {"CB", "CG", "CD"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "CYS": {"CB", "SG"},
}
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "ZN": 1.39, "MG": 1.73, "CA": 2.31, "FE": 1.56, "MN": 1.61,
    "CU": 1.40, "CO": 1.52, "NI": 1.63,
}
METALS = {"ZN", "MG", "CA", "FE", "MN", "CU", "CO", "NI", "CD", "HG"}


@dataclass(frozen=True, slots=True)
class RingFeature:
    atom_ids: tuple[int, ...]
    label: str
    source: str
    confidence: str


def analyze_structure(
    structure: NormalizedStructure,
    ligand_residue_name: str,
    ruleset: dict | None = None,
) -> list[InteractionRecord]:
    """Run all current native panVizPlus interaction detectors."""
    rules = ruleset or load_ruleset()
    records: list[InteractionRecord] = []
    records.extend(detect_conventional_hbonds(structure, ligand_residue_name, rules))
    records.extend(_detect_hydrophobic(structure, ligand_residue_name, rules))
    records.extend(_detect_salt_bridges(structure, ligand_residue_name, rules))
    records.extend(_detect_pi_interactions(structure, ligand_residue_name, rules))
    records.extend(_detect_halogen_bonds(structure, ligand_residue_name, rules))
    records.extend(_detect_metal_coordination(structure, ligand_residue_name, rules))
    records.extend(_detect_water_bridges(structure, ligand_residue_name, rules))
    records.extend(_detect_unfavorable_bumps(structure, ligand_residue_name, rules))
    return sorted(
        records,
        key=lambda r: (
            r.residue_number,
            r.chain_id or "",
            r.interaction_type,
            float(r.measurements.get("distance", r.measurements.get("donor_acceptor_distance", 999.0)) or 999.0),
        ),
    )


def _detect_hydrophobic(structure, ligand_name, rules):
    cutoff = float(rules["interactions"]["hydrophobic_contact"]["geometry"]["atom_distance_max"])
    ligand = [a for a in structure.ligand_atoms(ligand_name) if a.element.upper() in {"C", "S"}]
    protein = [
        a for a in structure.protein_atoms()
        if a.name.upper() in PROTEIN_HYDROPHOBIC_ATOMS.get(a.residue_name.upper(), set())
    ]
    best: dict[tuple[str, int, str], tuple[Atom, Atom, float]] = {}
    for la in ligand:
        for pa in protein:
            d = _distance(la, pa)
            if d <= cutoff:
                key = (pa.chain_id, pa.residue_number, pa.residue_name)
                if key not in best or d < best[key][2]:
                    best[key] = (la, pa, d)
    out = []
    for i, (_, (la, pa, d)) in enumerate(best.items(), 1):
        out.append(_record(
            f"HYD-{i:04d}", "hydrophobic_contact", la, pa, rules,
            {"distance": round(d, 3)},
            [CriterionResult("atom_distance", round(d, 3), "<=", cutoff, "angstrom", True)],
            chemistry_confidence="medium",
            chemistry_source="element_and_residue_template",
        ))
    return out


def _detect_salt_bridges(structure, ligand_name, rules):
    cutoff = float(rules["interactions"]["salt_bridge"]["geometry"]["charge_center_distance_max"])
    ligand_pos, ligand_neg = _ligand_charge_sites(structure, ligand_name)
    protein_pos = [a for a in structure.protein_atoms() if (a.residue_name.upper(), a.name.upper()) in PROTEIN_POSITIVE]
    protein_neg = [a for a in structure.protein_atoms() if (a.residue_name.upper(), a.name.upper()) in PROTEIN_NEGATIVE]
    pairs = [(lp, pn) for lp in ligand_pos for pn in protein_neg]
    pairs += [(ln, pp) for ln in ligand_neg for pp in protein_pos]
    out = []
    for la, pa in pairs:
        d = _distance(la, pa)
        if d <= cutoff:
            out.append(_record(
                f"SALT-{len(out)+1:04d}", "salt_bridge", la, pa, rules,
                {"distance": round(d, 3)},
                [CriterionResult("charge_center_distance", round(d, 3), "<=", cutoff, "angstrom", True)],
                chemistry_confidence="medium" if la.formal_charge else "low",
                chemistry_source="formal_charge_or_connectivity_inference",
            ))
    return _dedup_residue_type(out)


def _detect_pi_interactions(structure, ligand_name, rules):
    ligand_rings = _ligand_rings(structure, ligand_name)
    protein_rings = _protein_rings(structure)
    out: list[InteractionRecord] = []
    pp = rules["interactions"]["pi_pi"]
    center_max = float(pp["common_geometry"]["center_distance_max"])
    closest_max = float(pp["common_geometry"]["closest_atom_distance_max"])
    stacked_plane_max = float(pp["stacked"]["plane_angle_max"])
    tshape_plane_min = float(pp["t_shaped"]["plane_angle_min"])

    amap = structure.atom_map()
    for lr in ligand_rings:
        lc, ln = _ring_geometry([amap[x] for x in lr.atom_ids])
        if lc is None:
            continue
        for pr in protein_rings:
            pc, pn = _ring_geometry([amap[x] for x in pr.atom_ids])
            if pc is None:
                continue
            center = _vec_distance(lc, pc)
            if center > center_max:
                continue
            closest = min(_distance(amap[i], amap[j]) for i in lr.atom_ids for j in pr.atom_ids)
            if closest > closest_max:
                continue
            plane = _folded_angle(ln, pn)
            if plane <= stacked_plane_max:
                itype = "pi_pi_stacked"
                pass_angle = True
                threshold = stacked_plane_max
                comp = "<="
            elif plane >= tshape_plane_min:
                itype = "pi_pi_t_shaped"
                pass_angle = True
                threshold = tshape_plane_min
                comp = ">="
            else:
                continue
            pa = amap[pr.atom_ids[0]]
            la = amap[lr.atom_ids[0]]
            out.append(_record(
                f"PIPI-{len(out)+1:04d}", itype, la, pa, rules,
                {
                    "centroid_distance": round(center, 3),
                    "closest_atom_distance": round(closest, 3),
                    "plane_angle": round(plane, 2),
                    "ligand_ring": lr.label,
                    "protein_ring": pr.label,
                },
                [
                    CriterionResult("centroid_distance", round(center, 3), "<=", center_max, "angstrom", True),
                    CriterionResult("closest_atom_distance", round(closest, 3), "<=", closest_max, "angstrom", True),
                    CriterionResult("plane_angle", round(plane, 2), comp, threshold, "degree", pass_angle),
                ],
                ligand_site=lr.label,
                protein_site=pr.label,
                chemistry_confidence=lr.confidence,
                chemistry_source=lr.source,
            ))

    # pi-cation in either orientation
    pc_rule = rules["interactions"]["pi_cation"]
    cat_max = float(pc_rule["geometry"]["center_distance_max"])
    ligand_pos, _ = _ligand_charge_sites(structure, ligand_name)
    protein_pos = [a for a in structure.protein_atoms() if (a.residue_name.upper(), a.name.upper()) in PROTEIN_POSITIVE]

    for lr in ligand_rings:
        lc, _ = _ring_geometry([amap[x] for x in lr.atom_ids])
        if lc is None:
            continue
        for pa in protein_pos:
            d = _point_distance(lc, pa.coord)
            if d <= cat_max:
                out.append(_record(
                    f"PICAT-{len(out)+1:04d}", "pi_cation", amap[lr.atom_ids[0]], pa, rules,
                    {"center_distance": round(d, 3)},
                    [CriterionResult("center_distance", round(d, 3), "<=", cat_max, "angstrom", True)],
                    ligand_site=lr.label,
                    chemistry_confidence=lr.confidence,
                    chemistry_source=lr.source,
                ))
    for la in ligand_pos:
        for pr in protein_rings:
            pc, _ = _ring_geometry([amap[x] for x in pr.atom_ids])
            if pc is None:
                continue
            d = _point_distance(pc, la.coord)
            if d <= cat_max:
                pa = amap[pr.atom_ids[0]]
                out.append(_record(
                    f"PICAT-{len(out)+1:04d}", "pi_cation", la, pa, rules,
                    {"center_distance": round(d, 3)},
                    [CriterionResult("center_distance", round(d, 3), "<=", cat_max, "angstrom", True)],
                    protein_site=pr.label,
                    chemistry_confidence="medium" if la.formal_charge else "low",
                    chemistry_source="formal_charge_or_connectivity_inference",
                ))
    return _dedup_residue_type(out)


def _detect_halogen_bonds(structure, ligand_name, rules):
    rule = rules["interactions"]["halogen_bond"]
    distance_scale = float(rule["geometry"]["vdw_sum_scale_max"])
    angle_min = float(rule["geometry"]["C_X_A_angle_min"])
    ligand = structure.ligand_atoms(ligand_name)
    acceptors = [
        a for a in structure.protein_atoms()
        if a.element.upper() in {"O", "N", "S"}
    ]
    out = []
    for x in ligand:
        if x.element.upper() not in {"CL", "BR", "I"}:
            continue
        carbon_neighbors = [n for n in structure.neighbors(x.atom_id) if n.element.upper() == "C"]
        if not carbon_neighbors:
            continue
        carbon = carbon_neighbors[0]
        for a in acceptors:
            d = _distance(x, a)
            threshold = distance_scale * (VDW_RADII.get(x.element.upper(), 1.8) + VDW_RADII.get(a.element.upper(), 1.6))
            if d > threshold:
                continue
            angle = _angle(carbon.coord, x.coord, a.coord)
            if angle is None or angle < angle_min:
                continue
            out.append(_record(
                f"XB-{len(out)+1:04d}", "halogen_bond", x, a, rules,
                {"distance": round(d, 3), "C_X_A_angle": round(angle, 2)},
                [
                    CriterionResult("distance", round(d, 3), "<=", round(threshold, 3), "angstrom", True),
                    CriterionResult("C_X_A_angle", round(angle, 2), ">=", angle_min, "degree", True),
                ],
                chemistry_confidence="medium",
                chemistry_source="halogen_connectivity_geometry",
            ))
    return _dedup_residue_type(out)


def _detect_metal_coordination(structure, ligand_name, rules):
    cutoff = float(rules["interactions"]["metal_coordination"]["geometry"]["distance_max"])
    ligand_hetero = [a for a in structure.ligand_atoms(ligand_name) if a.element.upper() in {"N", "O", "S"}]
    protein_hetero = [a for a in structure.protein_atoms() if a.element.upper() in {"N", "O", "S"}]
    metals = [a for a in structure.atoms if a.element.upper() in METALS]
    out = []
    for metal in metals:
        ligand_hits = [(a, _distance(a, metal)) for a in ligand_hetero if _distance(a, metal) <= cutoff]
        protein_hits = [(a, _distance(a, metal)) for a in protein_hetero if _distance(a, metal) <= cutoff]
        for la, ld in ligand_hits:
            for pa, pd in protein_hits:
                out.append(_record(
                    f"METAL-{len(out)+1:04d}", "metal_coordination", la, pa, rules,
                    {
                        "ligand_metal_distance": round(ld, 3),
                        "protein_metal_distance": round(pd, 3),
                        "metal": f"{metal.element}{metal.atom_id}",
                    },
                    [
                        CriterionResult("ligand_metal_distance", round(ld, 3), "<=", cutoff, "angstrom", True),
                        CriterionResult("protein_metal_distance", round(pd, 3), "<=", cutoff, "angstrom", True),
                    ],
                    chemistry_confidence="high",
                    chemistry_source="element_identity_and_distance",
                ))
    return _dedup_residue_type(out)


def _detect_water_bridges(structure, ligand_name, rules):
    cutoff = float(rules["interactions"]["water_bridge"]["geometry"]["heavy_atom_distance_max"])
    protein_features, ligand_features = perceive_hbond_features(structure, ligand_name)
    amap = structure.atom_map()
    ligand_sites = [amap[f.atom_id] for f in ligand_features if f.kind in {"hydrogen_donor", "hydrogen_acceptor"}]
    protein_sites = [amap[f.atom_id] for f in protein_features if f.kind in {"hydrogen_donor", "hydrogen_acceptor"}]
    waters = [a for a in structure.water_atoms() if a.element.upper() == "O"]
    out = []
    for water in waters:
        lig = [(a, _distance(a, water)) for a in ligand_sites if _distance(a, water) <= cutoff]
        prot = [(a, _distance(a, water)) for a in protein_sites if _distance(a, water) <= cutoff]
        for la, ld in lig:
            for pa, pd in prot:
                out.append(_record(
                    f"WB-{len(out)+1:04d}", "water_bridge", la, pa, rules,
                    {
                        "ligand_water_distance": round(ld, 3),
                        "protein_water_distance": round(pd, 3),
                        "water_site": _site(water),
                    },
                    [
                        CriterionResult("ligand_water_distance", round(ld, 3), "<=", cutoff, "angstrom", True),
                        CriterionResult("protein_water_distance", round(pd, 3), "<=", cutoff, "angstrom", True),
                    ],
                    chemistry_confidence="medium",
                    chemistry_source="donor_acceptor_features_plus_water_geometry",
                ))
    return _dedup_residue_type(out)


def _detect_unfavorable_bumps(structure, ligand_name, rules):
    fraction = float(rules["interactions"]["unfavorable_vdw_bump"]["geometry"]["fraction_sum_vdw_radii_min"])
    ligand = [a for a in structure.ligand_atoms(ligand_name) if not a.is_hydrogen]
    protein = [a for a in structure.protein_atoms() if not a.is_hydrogen]
    best = {}
    for la in ligand:
        rl = VDW_RADII.get(la.element.upper())
        if rl is None:
            continue
        for pa in protein:
            rp = VDW_RADII.get(pa.element.upper())
            if rp is None:
                continue
            threshold = fraction * (rl + rp)
            d = _distance(la, pa)
            if d < threshold:
                key = (pa.chain_id, pa.residue_number, pa.residue_name)
                if key not in best or d < best[key][2]:
                    best[key] = (la, pa, d, threshold)
    out = []
    for i, (la, pa, d, threshold) in enumerate(best.values(), 1):
        out.append(_record(
            f"BUMP-{i:04d}", "unfavorable_vdw_bump", la, pa, rules,
            {"distance": round(d, 3), "minimum_allowed": round(threshold, 3)},
            [CriterionResult("distance", round(d, 3), "<", round(threshold, 3), "angstrom", True)],
            chemistry_confidence="high",
            chemistry_source="element_vdw_radii",
        ))
    return out


def _ligand_charge_sites(structure, ligand_name):
    atoms = structure.ligand_atoms(ligand_name)
    positive = [a for a in atoms if (a.formal_charge or 0) > 0]
    negative = [a for a in atoms if (a.formal_charge or 0) < 0]

    # Conservative fallback for common charged nitrogen environments.
    for a in atoms:
        if a.element.upper() == "N" and a not in positive:
            heavy = [n for n in structure.neighbors(a.atom_id) if not n.is_hydrogen]
            hydrogens = [n for n in structure.neighbors(a.atom_id) if n.is_hydrogen]
            if len(heavy) >= 4 or (len(heavy) == 3 and hydrogens):
                positive.append(a)

    # Carboxylate-like fallback is intentionally not assigned without explicit
    # charge/bond-order evidence; neutral acids must not be silently made negative.
    return positive, negative


def _protein_rings(structure) -> list[RingFeature]:
    by_res = {}
    for atom in structure.protein_atoms():
        by_res.setdefault((atom.chain_id, atom.residue_number, atom.residue_name), {})[atom.name.upper()] = atom
    out = []
    for (chain, resnum, resname), amap in by_res.items():
        for idx, names in enumerate(PROTEIN_AROMATIC_RINGS.get(resname.upper(), ())):
            if all(name in amap for name in names):
                atoms = tuple(amap[name].atom_id for name in names)
                out.append(RingFeature(
                    atoms,
                    f"{resname}{resnum}:{chain or '-'}:ring{idx+1}",
                    "protein_aromatic_residue_template",
                    "high",
                ))
    return out


def _ligand_rings(structure, ligand_name) -> list[RingFeature]:
    atoms = structure.ligand_atoms(ligand_name)
    allowed = {a.atom_id for a in atoms if not a.is_hydrogen}
    graph = {i: set() for i in allowed}
    for a, b in structure.bonds:
        if a in allowed and b in allowed:
            graph[a].add(b)
            graph[b].add(a)

    cycles = _cycles_5_6(graph)
    amap = structure.atom_map()
    out = []
    for idx, cycle in enumerate(cycles, 1):
        ring_atoms = [amap[i] for i in cycle]
        center, normal = _ring_geometry(ring_atoms)
        if center is None:
            continue
        # Planarity: all atoms within 0.20 A of fitted plane.
        dev = max(abs(np.dot(np.array(a.coord) - center, normal)) for a in ring_atoms)
        if dev > 0.20:
            continue
        hetero = sum(a.element.upper() in {"N", "O", "S"} for a in ring_atoms)
        if hetero > 2:
            confidence = "low"
        else:
            confidence = "medium"
        out.append(RingFeature(
            tuple(cycle),
            f"{ligand_name}:ring{idx}",
            "planar_5_6_member_cycle_from_connectivity",
            confidence,
        ))
    return out


def _cycles_5_6(graph):
    found = set()
    nodes = sorted(graph)
    for start in nodes:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > 6:
                continue
            for nxt in graph[node]:
                if nxt == start and len(path) in {5, 6}:
                    cyc = tuple(path)
                    found.add(_canonical_cycle(cyc))
                elif nxt not in path and nxt >= start:
                    stack.append((nxt, path + [nxt]))
    # Remove rings that are exact supersets of a smaller discovered ring.
    rings = sorted(found, key=lambda x: (len(x), x))
    unique = []
    for ring in rings:
        s = set(ring)
        if any(set(r) == s for r in unique):
            continue
        unique.append(ring)
    return unique


def _canonical_cycle(cycle):
    c = list(cycle)
    rots = []
    for seq in (c, list(reversed(c))):
        for i in range(len(seq)):
            rots.append(tuple(seq[i:] + seq[:i]))
    return min(rots)


def _ring_geometry(atoms: Iterable[Atom]):
    pts = np.array([a.coord for a in atoms], dtype=float)
    if len(pts) < 3:
        return None, None
    center = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - center)
    normal = vh[-1]
    n = np.linalg.norm(normal)
    if n < 1e-12:
        return None, None
    return center, normal / n


def _record(
    iid, itype, ligand_atom, protein_atom, rules, measurements, criteria,
    *, ligand_site=None, protein_site=None, chemistry_confidence, chemistry_source,
):
    return InteractionRecord(
        interaction_id=iid,
        interaction_type=itype,
        ligand_site=ligand_site or _site(ligand_atom),
        protein_site=protein_site or _site(protein_atom),
        residue_name=protein_atom.residue_name,
        residue_number=protein_atom.residue_number,
        chain_id=protein_atom.chain_id or None,
        detector="panVizPlus-native",
        detector_version=str(rules["metadata"]["version"]),
        ruleset=str(rules["metadata"]["id"]),
        criteria=criteria,
        measurements=measurements,
        metadata={
            "chemistry_confidence": chemistry_confidence,
            "chemistry_source": chemistry_source,
        },
    )


def _dedup_residue_type(records):
    best = {}
    for r in records:
        key = (r.interaction_type, r.chain_id, r.residue_number, r.residue_name)
        d = _primary_distance(r)
        if key not in best or d < _primary_distance(best[key]):
            best[key] = r
    return list(best.values())


def _primary_distance(record):
    for key in ("distance", "center_distance", "centroid_distance", "donor_acceptor_distance",
                "ligand_water_distance", "ligand_metal_distance"):
        value = record.measurements.get(key)
        if isinstance(value, (float, int)):
            return float(value)
    return 999.0


def _site(atom):
    return f"{atom.residue_name}:{atom.chain_id or '-'}:{atom.residue_number}:{atom.name}"


def _distance(a, b):
    return _point_distance(a.coord, b.coord)


def _point_distance(a, b):
    return sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _vec_distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _angle(a, vertex, c):
    v1 = np.asarray(a, dtype=float) - np.asarray(vertex, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(vertex, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    return degrees(acos(max(-1.0, min(1.0, cosv))))


def _folded_angle(n1, n2):
    cosv = abs(float(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))))
    return degrees(acos(max(-1.0, min(1.0, cosv))))
