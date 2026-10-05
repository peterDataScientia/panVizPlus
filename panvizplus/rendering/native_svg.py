"""Native SVG rendering for panVizPlus interaction records."""

from __future__ import annotations

from html import escape
from math import cos, pi, sin

import numpy as np

from panvizplus.chemistry.models import NormalizedStructure
from panvizplus.chemistry.rdkit_layer import build_ligand_chemistry, ligand_2d_coordinates
from panvizplus.interactions.models import InteractionRecord


COLORS = {
    "conventional_hbond": "#2457E6",
    "hydrophobic_contact": "#666666",
    "salt_bridge": "#D000A8",
    "pi_pi_stacked": "#16823B",
    "pi_pi_t_shaped": "#16823B",
    "pi_cation": "#D97706",
    "halogen_bond": "#7557C8",
    "metal_coordination": "#A55A00",
    "water_bridge": "#1289A7",
    "unfavorable_vdw_bump": "#C62828",
}
LABELS = {
    "conventional_hbond": "Hydrogen bond",
    "hydrophobic_contact": "Hydrophobic",
    "salt_bridge": "Salt bridge",
    "pi_pi_stacked": "π–π stacked",
    "pi_pi_t_shaped": "π–π T-shaped",
    "pi_cation": "π–cation",
    "halogen_bond": "Halogen bond",
    "metal_coordination": "Metal coordination",
    "water_bridge": "Water bridge",
    "unfavorable_vdw_bump": "Unfavorable contact",
}
DASH = {
    "conventional_hbond": "7 5",
    "hydrophobic_contact": "3 5",
    "salt_bridge": "9 4",
    "pi_pi_stacked": "7 4",
    "pi_pi_t_shaped": "7 4",
    "pi_cation": "8 4",
    "halogen_bond": "6 4",
    "metal_coordination": "4 3",
    "water_bridge": "5 4",
    "unfavorable_vdw_bump": "2 3",
}


def render_interaction_svg(
    structure: NormalizedStructure,
    ligand_selector: str,
    records: list[InteractionRecord],
    width: int = 1200,
    height: int = 820,
    ligand_net_charge: int | None = None,
) -> str:
    ligand = [a for a in structure.ligand_atoms(ligand_selector) if not a.is_hydrogen]
    if not ligand:
        raise ValueError("Selected ligand has no heavy atoms.")

    chemistry = build_ligand_chemistry(
        structure, ligand_selector, net_charge=ligand_net_charge
    )
    coords = _project_ligand_rdkit(chemistry, ligand, width, height)
    atom_by_name = {a.name: a for a in ligand}
    ligand_center = (
        float(np.mean([xy[0] for xy in coords.values()])),
        float(np.mean([xy[1] for xy in coords.values()])),
    )

    residues = []
    seen = set()
    for rec in records:
        key = (rec.residue_name, rec.residue_number, rec.chain_id or "-")
        if key not in seen:
            seen.add(key)
            residues.append(key)

    res_xy = {}
    radius_x = max(290.0, width * 0.34)
    radius_y = max(230.0, height * 0.34)
    for i, key in enumerate(residues):
        angle = -pi / 2 + (2 * pi * i / max(1, len(residues)))
        res_xy[key] = (
            width / 2 + radius_x * cos(angle),
            height / 2 + radius_y * sin(angle),
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif}.res{font-weight:700;font-size:18px}.atom{font-size:13px;font-weight:700}.dist{font-size:12px;fill:#222}</style>',
    ]

    ligand_ids = {a.atom_id for a in ligand}
    amap = structure.atom_map()
    for left, right in structure.bonds:
        if left in ligand_ids and right in ligand_ids:
            x1, y1 = coords[left]
            x2, y2 = coords[right]
            order = chemistry.bond_orders.get(tuple(sorted((left, right))), 1.0)
            parts.extend(_bond_svg(x1, y1, x2, y2, order))

    for rec in records:
        key = (rec.residue_name, rec.residue_number, rec.chain_id or "-")
        rx, ry = res_xy[key]
        lx, ly = _ligand_anchor(rec, atom_by_name, coords, ligand_center)
        color = COLORS.get(rec.interaction_type, "#555")
        dash = DASH.get(rec.interaction_type, "6 4")
        parts.append(
            f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{rx:.1f}" y2="{ry:.1f}" '
            f'stroke="{color}" stroke-width="2.2" stroke-dasharray="{dash}"/>'
        )
        d = _primary_distance(rec)
        if d is not None:
            mx, my = (lx + rx) / 2, (ly + ry) / 2
            parts.append(f'<rect x="{mx-24:.1f}" y="{my-10:.1f}" width="48" height="18" rx="5" fill="white" opacity=".88"/>')
            parts.append(f'<text class="dist" x="{mx:.1f}" y="{my+3:.1f}" text-anchor="middle">{d:.2f} Å</text>')

    for atom in ligand:
        x, y = coords[atom.atom_id]
        el = atom.element.upper()
        fill = {
            "O": "#D33", "N": "#2457E6", "S": "#D59B00", "P": "#E87500",
            "F": "#3B9D55", "CL": "#3B9D55", "BR": "#8C4A2F", "I": "#6D4A8B",
        }.get(el, "#222")
        r = 6 if el == "C" else 8
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="white" stroke-width="1.2"/>')
        if el != "C":
            parts.append(f'<text class="atom" x="{x:.1f}" y="{y-11:.1f}" text-anchor="middle">{escape(atom.element)}</text>')

    for key, (x, y) in res_xy.items():
        resname, resnum, chain = key
        label = f"{resname}{resnum}:{chain}"
        parts.append(f'<rect x="{x-48:.1f}" y="{y-18:.1f}" width="96" height="36" rx="18" fill="#F7FAFE" stroke="#7890A8" stroke-width="1.4"/>')
        parts.append(f'<text class="res" x="{x:.1f}" y="{y+6:.1f}" text-anchor="middle">{escape(label)}</text>')

    legend_types = []
    for rec in records:
        if rec.interaction_type not in legend_types:
            legend_types.append(rec.interaction_type)
    lx, ly = 22, height - 24 - 22 * max(0, len(legend_types) - 1)
    for i, itype in enumerate(legend_types):
        y = ly + i * 22
        color = COLORS.get(itype, "#555")
        dash = DASH.get(itype, "6 4")
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+28}" y2="{y}" stroke="{color}" stroke-width="2.2" stroke-dasharray="{dash}"/>')
        parts.append(f'<text x="{lx+36}" y="{y+4}" font-size="12">{escape(LABELS.get(itype, itype))}</text>')

    parts.append('</svg>')
    return "".join(parts)



def _project_ligand_rdkit(chemistry, atoms, width, height):
    raw = ligand_2d_coordinates(chemistry)
    ids = [a.atom_id for a in atoms if a.atom_id in raw]
    pts = np.asarray([raw[i] for i in ids], dtype=float)
    if pts.size == 0:
        return {a.atom_id: (width / 2, height / 2) for a in atoms}

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    xr = max(float(xmax - xmin), 1.0)
    yr = max(float(ymax - ymin), 1.0)
    max_w, max_h = width * 0.38, height * 0.46
    scale = min(max_w / xr, max_h / yr)
    cx, cy = width / 2, height / 2
    mean = pts.mean(axis=0)
    return {
        a.atom_id: (
            cx + (raw[a.atom_id][0] - mean[0]) * scale,
            cy - (raw[a.atom_id][1] - mean[1]) * scale,
        )
        for a in atoms
        if a.atom_id in raw
    }


def _bond_svg(x1, y1, x2, y2, order):
    base = f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#333" stroke-width="2.1"/>'
    if order < 1.5:
        return [base]
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ox, oy = -dy / length * 3.0, dx / length * 3.0
    second = (
        f'<line x1="{x1+ox:.1f}" y1="{y1+oy:.1f}" '
        f'x2="{x2+ox:.1f}" y2="{y2+oy:.1f}" stroke="#333" stroke-width="1.6"/>'
    )
    if order >= 2.5:
        third = (
            f'<line x1="{x1-ox:.1f}" y1="{y1-oy:.1f}" '
            f'x2="{x2-ox:.1f}" y2="{y2-oy:.1f}" stroke="#333" stroke-width="1.4"/>'
        )
        return [base, second, third]
    return [base, second]

def _ligand_anchor(rec, atom_by_name, coords, center):
    site = str(rec.ligand_site)
    atom_name = site.split(":")[-1]
    atom = atom_by_name.get(atom_name)
    if atom is not None and atom.atom_id in coords:
        return coords[atom.atom_id]
    return center


def _primary_distance(rec):
    for key in (
        "distance", "donor_acceptor_distance", "center_distance", "centroid_distance",
        "ligand_water_distance", "ligand_metal_distance",
    ):
        value = rec.measurements.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None
