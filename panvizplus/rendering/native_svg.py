"""Publication-oriented SVG rendering for panVizPlus interaction records."""

from __future__ import annotations

from html import escape
from math import cos, pi, sin
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from panvizplus.chemistry.models import NormalizedStructure
from panvizplus.chemistry.rdkit_layer import LigandChemistry, build_ligand_chemistry
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

    panel_w = int(min(max(width * 0.48, 430), 650))
    panel_h = int(min(max(height * 0.54, 340), 520))
    panel_x = (width - panel_w) / 2
    panel_y = (height - panel_h) / 2

    ligand_svg, local_coords = _publication_ligand_svg(chemistry, panel_w, panel_h)
    coords = {
        chemistry.rd_idx_to_atom_id[idx]: (panel_x + xy[0], panel_y + xy[1])
        for idx, xy in local_coords.items()
        if idx in chemistry.rd_idx_to_atom_id
    }
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

    res_xy = _residue_positions(residues, width, height, panel_w, panel_h)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>'
        'text{font-family:Arial,Helvetica,sans-serif}'
        '.res{font-weight:700;font-size:17px;fill:#15243A}'
        '.dist{font-size:12px;fill:#27384A}'
        '.legend{font-size:12px;fill:#263649}'
        '</style>',
    ]

    # Interaction geometry is deliberately drawn before the molecule so that
    # chemical bonds and atom labels remain visually dominant and unobscured.
    for rec in records:
        key = (rec.residue_name, rec.residue_number, rec.chain_id or "-")
        rx, ry = res_xy[key]
        lx, ly = _ligand_anchor(
            rec, atom_by_name, coords, ligand_center, chemistry
        )
        color = COLORS.get(rec.interaction_type, "#555")
        dash = DASH.get(rec.interaction_type, "6 4")
        ex, ey = _stop_before_residue(lx, ly, rx, ry, 52.0)
        parts.append(
            f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="{color}" stroke-width="2.0" stroke-dasharray="{dash}" '
            'stroke-linecap="round"/>'
        )
        d = _primary_distance(rec)
        if d is not None:
            # Keep numeric labels away from the ligand core.
            tx = lx + 0.64 * (ex - lx)
            ty = ly + 0.64 * (ey - ly)
            parts.append(
                f'<rect x="{tx-24:.1f}" y="{ty-10:.1f}" width="48" height="18" '
                'rx="4" fill="white" stroke="#E4E8EE" stroke-width=".7"/>'
            )
            parts.append(
                f'<text class="dist" x="{tx:.1f}" y="{ty+4:.1f}" '
                f'text-anchor="middle">{d:.2f} Å</text>'
            )

    # Embed the RDKit chemical drawing as vector paths, not a raster image and
    # not the previous node-and-edge graph representation.
    parts.append(
        f'<g transform="translate({panel_x:.1f},{panel_y:.1f})">{ligand_svg}</g>'
    )

    for key, (x, y) in res_xy.items():
        resname, resnum, chain = key
        label = f"{resname}{resnum}:{chain}"
        parts.append(
            f'<rect x="{x-50:.1f}" y="{y-18:.1f}" width="100" height="36" rx="18" '
            'fill="white" stroke="#7B8FA6" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text class="res" x="{x:.1f}" y="{y+6:.1f}" '
            f'text-anchor="middle">{escape(label)}</text>'
        )

    legend_types = []
    for rec in records:
        if rec.interaction_type not in legend_types:
            legend_types.append(rec.interaction_type)
    lx, ly = 24, height - 24 - 22 * max(0, len(legend_types) - 1)
    for i, itype in enumerate(legend_types):
        y = ly + i * 22
        color = COLORS.get(itype, "#555")
        dash = DASH.get(itype, "6 4")
        parts.append(
            f'<line x1="{lx}" y1="{y}" x2="{lx+28}" y2="{y}" '
            f'stroke="{color}" stroke-width="2.0" stroke-dasharray="{dash}"/>'
        )
        parts.append(
            f'<text class="legend" x="{lx+36}" y="{y+4}">'
            f'{escape(LABELS.get(itype, itype))}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _publication_ligand_svg(
    chemistry: LigandChemistry,
    width: int,
    height: int,
) -> tuple[str, dict[int, tuple[float, float]]]:
    """Return publication-style skeletal SVG and RDKit-index draw coordinates."""
    mol = Chem.Mol(chemistry.mol)
    mol.UpdatePropertyCache(strict=False)

    # Use a true 2D chemical layout. This replaces the previous black-node graph.
    rdDepictor.Compute2DCoords(mol, canonOrient=True)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.035
    opts.bondLineWidth = 1.8
    opts.minFontSize = 13
    opts.maxFontSize = 22
    opts.addStereoAnnotation = True
    opts.explicitMethyl = False
    opts.multipleBondOffset = 0.16

    # PrepareAndDrawMolecule gives RDKit control over aromatic/double-bond
    # placement, heteroatom labels, stereobonds, and canonical ring depiction.
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)

    draw_coords: dict[int, tuple[float, float]] = {}
    for idx in range(mol.GetNumAtoms()):
        p = drawer.GetDrawCoords(idx)
        draw_coords[idx] = (float(p.x), float(p.y))

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    inner = _strip_outer_svg(svg)
    return inner, draw_coords


def _strip_outer_svg(svg: str) -> str:
    start = re.search(r"<svg\b[^>]*>", svg, flags=re.I | re.S)
    if not start:
        return svg
    inner = svg[start.end():]
    inner = re.sub(r"</svg>\s*$", "", inner, flags=re.I | re.S)
    # The RDKit drawing can contain a white background rectangle. The parent
    # figure already owns the background, so remove only that canvas rectangle.
    inner = re.sub(
        r"<rect[^>]*style=['\"][^'\"]*fill:#FFFFFF[^'\"]*['\"][^>]*/>",
        "",
        inner,
        count=1,
        flags=re.I,
    )
    return inner


def _residue_positions(residues, width, height, panel_w, panel_h):
    positions = {}
    if not residues:
        return positions
    radius_x = max(panel_w * 0.72, width * 0.35)
    radius_y = max(panel_h * 0.72, height * 0.35)
    for i, key in enumerate(residues):
        angle = -pi / 2 + (2 * pi * i / len(residues))
        positions[key] = (
            width / 2 + radius_x * cos(angle),
            height / 2 + radius_y * sin(angle),
        )
    return positions


def _ligand_anchor(rec, atom_by_name, coords, center, chemistry):
    site = str(rec.ligand_site)
    atom_name = site.split(":")[-1]
    atom = atom_by_name.get(atom_name)
    if atom is not None and atom.atom_id in coords:
        return coords[atom.atom_id]

    ring_match = re.search(r"aromatic_ring(\d+)", site)
    if ring_match:
        ring_index = int(ring_match.group(1)) - 1
        if 0 <= ring_index < len(chemistry.aromatic_rings):
            ring = chemistry.aromatic_rings[ring_index]
            pts = [coords[a] for a in ring if a in coords]
            if pts:
                return (
                    float(np.mean([p[0] for p in pts])),
                    float(np.mean([p[1] for p in pts])),
                )
    return center


def _stop_before_residue(x1, y1, x2, y2, radius):
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
    return x2 - dx / length * radius, y2 - dy / length * radius


def _primary_distance(rec):
    for key in (
        "distance", "donor_acceptor_distance", "center_distance", "centroid_distance",
        "ligand_water_distance", "ligand_metal_distance",
    ):
        value = rec.measurements.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None
