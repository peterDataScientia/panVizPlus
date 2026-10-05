"""PanViz-style publication SVG rendering for native panVizPlus records.

The scientific interaction engine is independent of this module.  The renderer
adapts the established PanViz visual grammar: chemical skeletal ligand,
interaction-specific dashed connectors, local residue anchors, subtle leaders,
60 px glass residue bubbles, magenta distance labels, and a centered boxed
publication legend.
"""

from __future__ import annotations

from collections import Counter
from html import escape
from math import cos, hypot, pi, sin
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

from panvizplus.chemistry.models import NormalizedStructure
from panvizplus.chemistry.rdkit_layer import LigandChemistry, build_ligand_chemistry
from panvizplus.interactions.models import InteractionRecord


BUBBLE_COLOR = "#0AFFEF"
DISTANCE_COLOR = "#D100A0"

# PanViz publication semantic colors.
COLORS = {
    "hydrophobic_contact": "#595959",
    "conventional_hbond": "#0000E0",
    "water_bridge": "#1596B8",
    "salt_bridge": "#D900B0",
    "pi_pi_stacked": "#008500",
    "pi_pi_t_shaped": "#008500",
    "pi_cation": "#E08000",
    "halogen_bond": "#7A5CC7",
    "metal_coordination": "#A45700",
    "unfavorable_vdw_bump": "#C62828",
}
LABELS = {
    "hydrophobic_contact": "Hydrophobic contact",
    "conventional_hbond": "Hydrogen bond",
    "water_bridge": "Water bridge",
    "salt_bridge": "Salt bridge",
    "pi_pi_stacked": "π–π stacked",
    "pi_pi_t_shaped": "π–π T-shaped",
    "pi_cation": "π–Cation",
    "halogen_bond": "Halogen bond",
    "metal_coordination": "Metal coordination",
    "unfavorable_vdw_bump": "Unfavorable contact",
}
INTERACTION_ORDER = (
    "hydrophobic_contact",
    "conventional_hbond",
    "water_bridge",
    "salt_bridge",
    "pi_pi_stacked",
    "pi_pi_t_shaped",
    "pi_cation",
    "halogen_bond",
    "metal_coordination",
    "unfavorable_vdw_bump",
)
DASH = {
    "hydrophobic_contact": "9 5",
    "conventional_hbond": "9 5",
    "water_bridge": "3 3",
    "salt_bridge": "9 5",
    "pi_pi_stacked": "9 5",
    "pi_pi_t_shaped": "9 5",
    "pi_cation": "9 5",
    "halogen_bond": "6 4",
    "metal_coordination": "2 3",
    "unfavorable_vdw_bump": "2 3",
}
LINE_SHIFT = {
    "hydrophobic_contact": 0.0,
    "conventional_hbond": 6.0,
    "pi_pi_stacked": 10.0,
    "pi_pi_t_shaped": 10.0,
    "pi_cation": -10.0,
    "salt_bridge": -5.0,
    "water_bridge": 4.0,
    "halogen_bond": -8.0,
    "metal_coordination": 8.0,
    "unfavorable_vdw_bump": -4.0,
}


def render_interaction_svg(
    structure: NormalizedStructure,
    ligand_selector: str,
    records: list[InteractionRecord],
    width: int = 1200,
    height: int = 820,
    ligand_net_charge: int | None = None,
) -> str:
    ligand = [
        a for a in structure.ligand_atoms(ligand_selector)
        if not a.is_hydrogen
    ]
    if not ligand:
        raise ValueError("Selected ligand has no heavy atoms.")

    chemistry = build_ligand_chemistry(
        structure,
        ligand_selector,
        net_charge=ligand_net_charge,
    )

    legend = _legend_layout(records, width, height)
    legend_reserve = legend["h"] + 30.0 if legend else 10.0
    content_bottom = max(220.0, float(height) - legend_reserve)

    panel_w = int(min(max(width * 0.54, 470), 720))
    panel_h = int(min(max(content_bottom * 0.58, 350), 560))
    panel_x = (width - panel_w) / 2.0
    panel_y = max(20.0, (content_bottom - panel_h) / 2.0)

    ligand_svg, local_coords = _publication_ligand_svg(
        chemistry,
        panel_w,
        panel_h,
    )
    coords = {
        chemistry.rd_idx_to_atom_id[idx]: (
            panel_x + xy[0],
            panel_y + xy[1],
        )
        for idx, xy in local_coords.items()
        if idx in chemistry.rd_idx_to_atom_id
    }
    if not coords:
        raise ValueError("Unable to place ligand atoms in the publication scene.")

    atom_by_name = {a.name: a for a in ligand}
    ligand_center = (
        float(np.mean([xy[0] for xy in coords.values()])),
        float(np.mean([xy[1] for xy in coords.values()])),
    )

    residue_layout = _residue_layout(
        records,
        atom_by_name,
        coords,
        ligand_center,
        chemistry,
        width,
        content_bottom,
    )

    pair_counts = Counter(
        (
            (r.residue_name, r.residue_number, r.chain_id or "-"),
            str(r.ligand_site),
        )
        for r in records
    )

    interaction_lines = []
    for rec in records:
        key = (rec.residue_name, rec.residue_number, rec.chain_id or "-")
        layout = residue_layout.get(key)
        if layout is None:
            continue
        x1, y1 = _ligand_anchor(
            rec,
            atom_by_name,
            coords,
            ligand_center,
            chemistry,
        )
        x2, y2 = layout["anchor"]
        dx, dy = x2 - x1, y2 - y1
        length = hypot(dx, dy)
        if length < 1e-9:
            continue
        shift = LINE_SHIFT.get(rec.interaction_type, 0.0)
        ox = (-dy / length) * shift
        oy = (dx / length) * shift
        multiplicity = max(1, pair_counts[(key, str(rec.ligand_site))])
        interaction_lines.append({
            "record": rec,
            "x1": x1 + ox,
            "y1": y1 + oy,
            "x2": x2,
            "y2": y2,
            "distance_x1": x1,
            "distance_y1": y1,
            "distance_x2": x2,
            "distance_y2": y2,
            "width": 3.0 * multiplicity,
        })

    bubble_boxes = [
        (v["x"], v["y"], 60.0, 60.0)
        for v in residue_layout.values()
    ]
    distance_labels = _distance_label_layout(
        interaction_lines,
        bubble_boxes,
        width,
        content_bottom,
    )

    residue_font_size = max(17, min(23, int(min(width, height) * 0.025)))
    distance_font_size = max(14, min(18, int(min(width, height) * 0.018)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-renderer="panviz-publication">',
        '<defs>',
        '<filter id="pv-bubble-shadow" x="-30%" y="-35%" width="170%" height="180%">'
        '<feDropShadow dx="0" dy="3" stdDeviation="3.2" flood-color="#1f2937" flood-opacity=".22"/>'
        '</filter>',
        '<radialGradient id="pv-bubble-gradient" cx="30%" cy="24%" r="76%" fx="28%" fy="22%">'
        '<stop offset="0%" stop-color="#FFFFFF" stop-opacity=".98"/>'
        '<stop offset="10%" stop-color="#D3FFFC" stop-opacity=".96"/>'
        '<stop offset="34%" stop-color="#58FFF4" stop-opacity=".92"/>'
        f'<stop offset="70%" stop-color="{BUBBLE_COLOR}" stop-opacity=".88"/>'
        '<stop offset="100%" stop-color="#08CCBF" stop-opacity=".92"/>'
        '</radialGradient>',
        '</defs>',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>'
        'text{font-family:Arial,Helvetica,sans-serif}'
        '.pv-distance{font-weight:700}'
        '.pv-legend{font-weight:700;fill:#111111}'
        '</style>',
    ]

    # Non-covalent interaction layer.  The ligand is painted afterwards so the
    # molecular structure remains the visually dominant scientific layer.
    parts.append('<g id="pv-interactions">')
    for item in interaction_lines:
        rec = item["record"]
        color = COLORS.get(rec.interaction_type, "#595959")
        dash = DASH.get(rec.interaction_type, "9 5")
        parts.append(
            f'<line class="pv-interaction pv-{escape(rec.interaction_type)}" '
            f'x1="{item["x1"]:.1f}" y1="{item["y1"]:.1f}" '
            f'x2="{item["x2"]:.1f}" y2="{item["y2"]:.1f}" '
            f'stroke="{color}" stroke-width="{item["width"]:.1f}" '
            f'stroke-dasharray="{dash}" stroke-linecap="round" opacity=".92"/>'
        )
    parts.append('</g>')

    # PanViz leaders connect the scientific residue anchor to a displaced
    # presentation bubble; they are deliberately neutral and subtle.
    parts.append('<g id="pv-residue-leaders">')
    for layout in residue_layout.values():
        leader = _leader_geometry(layout)
        if leader is None:
            continue
        parts.append(
            f'<line class="pv-residue-leader" '
            f'x1="{leader[0]:.1f}" y1="{leader[1]:.1f}" '
            f'x2="{leader[2]:.1f}" y2="{leader[3]:.1f}" '
            'stroke="#000000" stroke-width="1.1" opacity=".30" '
            'stroke-linecap="round"/>'
        )
    parts.append('</g>')

    parts.append(
        f'<g id="pv-ligand" transform="translate({panel_x:.1f},{panel_y:.1f})">'
        f'{ligand_svg}</g>'
    )

    # Collision-aware PanViz distance labels: bold magenta text with an opaque
    # white knockout rather than pill-style badges.
    parts.append('<g id="pv-distance-labels">')
    for dl in distance_labels:
        parts.append(
            f'<rect x="{dl["x"]-dl["w"]/2-4:.1f}" '
            f'y="{dl["y"]-dl["h"]/2-3:.1f}" '
            f'width="{dl["w"]+8:.1f}" height="{dl["h"]+6:.1f}" '
            'fill="#FFFFFF" fill-opacity=".96"/>'
        )
        parts.append(
            f'<text class="pv-distance" x="{dl["x"]:.1f}" '
            f'y="{dl["y"]+distance_font_size*0.34:.1f}" '
            f'text-anchor="middle" font-size="{distance_font_size}" '
            f'fill="{DISTANCE_COLOR}">{escape(dl["text"])}</text>'
        )
    parts.append('</g>')

    # Established PanViz 60 px glass-bubble residue nodes.
    parts.append('<g id="pv-residue-labels">')
    for key, layout in residue_layout.items():
        resname, resnum, chain = key
        x, y = layout["x"], layout["y"]
        r = 30.0
        parts.extend([
            f'<circle class="pv-residue-bubble" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
            'fill="url(#pv-bubble-gradient)" fill-opacity=".98" '
            'stroke="#08C2B6" stroke-width="1.25" filter="url(#pv-bubble-shadow)"/>',
            f'<ellipse cx="{x-r*.30:.1f}" cy="{y-r*.36:.1f}" '
            f'rx="{r*.34:.1f}" ry="{r*.19:.1f}" fill="#FFFFFF" opacity=".52"/>',
            f'<ellipse cx="{x:.1f}" cy="{y+r*.36:.1f}" '
            f'rx="{r*.66:.1f}" ry="{r*.29:.1f}" fill="#FFFFFF" opacity=".08"/>',
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r*.92:.1f}" fill="none" '
            'stroke="#FFFFFF" stroke-width=".8" opacity=".22"/>',
        ])
        nfs = min(residue_font_size * 0.78, max(8.0, (r) * 0.42))
        parts.append(
            f'<text class="pv-residue-text" x="{x:.1f}" y="{y-2:.1f}" '
            f'text-anchor="middle" font-size="{nfs:.1f}" font-weight="700" '
            'fill="#000000">'
            f'<tspan x="{x:.1f}" dy="0">{escape(str(resname).upper())}</tspan>'
            f'<tspan x="{x:.1f}" dy="{nfs*1.02:.1f}">{escape(str(resnum))}:{escape(str(chain))}</tspan>'
            '</text>'
        )
    parts.append('</g>')

    if legend:
        parts.extend(_render_legend(legend))

    parts.append("</svg>")
    return "".join(parts)


def _publication_ligand_svg(
    chemistry: LigandChemistry,
    width: int,
    height: int,
) -> tuple[str, dict[int, tuple[float, float]]]:
    """Draw a true chemical skeletal ligand using RDKit vector output."""
    mol = Chem.Mol(chemistry.mol)
    mol.UpdatePropertyCache(strict=False)
    rdDepictor.Compute2DCoords(mol, canonOrient=True)

    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return _skeletal_svg_fallback(mol, width, height)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.035
    opts.bondLineWidth = 2.0
    opts.minFontSize = 14
    opts.maxFontSize = 19
    opts.addStereoAnnotation = True
    opts.explicitMethyl = False
    opts.multipleBondOffset = 0.18

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)

    draw_coords: dict[int, tuple[float, float]] = {}
    for idx in range(mol.GetNumAtoms()):
        p = drawer.GetDrawCoords(idx)
        draw_coords[idx] = (float(p.x), float(p.y))

    drawer.FinishDrawing()
    inner = _strip_outer_svg(drawer.GetDrawingText())
    return inner, draw_coords


def _skeletal_svg_fallback(
    mol: Chem.Mol,
    width: int,
    height: int,
) -> tuple[str, dict[int, tuple[float, float]]]:
    """Dependency-safe PanViz-like skeletal vector fallback."""
    conf = mol.GetConformer()
    raw = {
        idx: (
            float(conf.GetAtomPosition(idx).x),
            float(conf.GetAtomPosition(idx).y),
        )
        for idx in range(mol.GetNumAtoms())
    }
    heavy = [
        idx for idx, atom in enumerate(mol.GetAtoms())
        if atom.GetAtomicNum() != 1
    ]
    pts = np.asarray([raw[idx] for idx in heavy], dtype=float)
    if pts.size == 0:
        return "", {}

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    xr = max(float(xmax - xmin), 1.0)
    yr = max(float(ymax - ymin), 1.0)
    scale = min((width * 0.86) / xr, (height * 0.80) / yr)
    mean = pts.mean(axis=0)
    coords = {
        idx: (
            width / 2 + (raw[idx][0] - mean[0]) * scale,
            height / 2 - (raw[idx][1] - mean[1]) * scale,
        )
        for idx in raw
    }

    parts = ['<g id="pv-ligand-fallback">']
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if (
            mol.GetAtomWithIdx(a).GetAtomicNum() == 1
            or mol.GetAtomWithIdx(b).GetAtomicNum() == 1
        ):
            continue
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        parts.extend(
            _fallback_bond_svg(
                x1,
                y1,
                x2,
                y2,
                float(bond.GetBondTypeAsDouble()),
            )
        )

    element_colours = {
        7: "#0000CC",
        8: "#D90000",
        15: "#D96600",
        16: "#B78A00",
        5: "#E57373",
        9: "#159647",
        17: "#159647",
        35: "#8C4A2F",
        53: "#6D4A8B",
    }
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() in {1, 6}:
            continue
        x, y = coords[idx]
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        charge_text = ""
        if charge > 0:
            charge_text = "+" if charge == 1 else f"{charge}+"
        elif charge < 0:
            charge_text = "−" if charge == -1 else f"{abs(charge)}−"
        colour = element_colours.get(atom.GetAtomicNum(), "#000000")
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8.5" fill="#FFFFFF" opacity=".95"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y+5:.1f}" text-anchor="middle" '
            'font-family="Arial,Helvetica,sans-serif" font-size="16" '
            f'font-weight="700" fill="{colour}">{escape(symbol)}'
            + (
                f'<tspan baseline-shift="super" font-size="10">{escape(charge_text)}</tspan>'
                if charge_text else ""
            )
            + '</text>'
        )
    parts.append('</g>')
    return "".join(parts), coords


def _fallback_bond_svg(x1, y1, x2, y2, order):
    dx, dy = x2 - x1, y2 - y1
    length = max(hypot(dx, dy), 1e-9)
    ox, oy = -dy / length * 3.0, dx / length * 3.0

    def line(ax, ay, bx, by):
        return (
            f'<line class="pv-ligand-bond" x1="{ax:.1f}" y1="{ay:.1f}" '
            f'x2="{bx:.1f}" y2="{by:.1f}" stroke="#000000" '
            'stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    if 1.35 <= order < 2.5:
        return [
            line(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
            line(x1 - ox, y1 - oy, x2 - ox, y2 - oy),
        ]
    if order >= 2.5:
        return [
            line(x1, y1, x2, y2),
            line(x1 + ox * 1.5, y1 + oy * 1.5, x2 + ox * 1.5, y2 + oy * 1.5),
            line(x1 - ox * 1.5, y1 - oy * 1.5, x2 - ox * 1.5, y2 - oy * 1.5),
        ]
    return [line(x1, y1, x2, y2)]


def _strip_outer_svg(svg: str) -> str:
    start = re.search(r"<svg\b[^>]*>", svg, flags=re.I | re.S)
    if not start:
        return svg
    inner = svg[start.end():]
    inner = re.sub(r"</svg>\s*$", "", inner, flags=re.I | re.S)
    inner = re.sub(
        r"<rect[^>]*style=['\"][^'\"]*fill:#FFFFFF[^'\"]*['\"][^>]*/>",
        "",
        inner,
        count=1,
        flags=re.I,
    )
    return inner


def _residue_layout(
    records,
    atom_by_name,
    coords,
    ligand_center,
    chemistry,
    width,
    content_bottom,
):
    if not records:
        return {}

    grouped = {}
    for rec in records:
        key = (rec.residue_name, rec.residue_number, rec.chain_id or "-")
        grouped.setdefault(key, []).append(rec)

    atom_points = list(coords.values())
    cx, cy = ligand_center
    node_size = 60.0
    margin = 14.0

    anchors = []
    for key, recs in grouped.items():
        pts = [
            _ligand_anchor(rec, atom_by_name, coords, ligand_center, chemistry)
            for rec in recs
        ]
        site_x = float(np.mean([p[0] for p in pts]))
        site_y = float(np.mean([p[1] for p in pts]))
        ux, uy = _outward_unit(key, site_x, site_y, ligand_center)
        # This is the scientific residue-side anchor used by interaction lines.
        anchor_offset = max(58.0, min(82.0, min(width, content_bottom) * 0.085))
        anchor_x = site_x + ux * anchor_offset
        anchor_y = site_y + uy * anchor_offset
        anchors.append((key, site_x, site_y, anchor_x, anchor_y, ux, uy))

    anchors.sort(
        key=lambda z: hypot(z[1] - cx, z[2] - cy)
    )

    placed_boxes = []
    placed_leaders = []
    out = {}

    for key, site_x, site_y, anchor_x, anchor_y, ux, uy in anchors:
        nx, ny = -uy, ux
        candidates = []
        for radial in (20, 28, 38, 50, 64, 80, 98):
            for tangential in (0, 8, -8, 16, -16, 28, -28):
                candidates.append((
                    anchor_x + ux * radial + nx * tangential,
                    anchor_y + uy * radial + ny * tangential,
                ))
        for radial in (24, 36, 52):
            candidates.append((
                anchor_x - ux * radial,
                anchor_y - uy * radial,
            ))

        chosen = None
        best = float("inf")
        for px, py in candidates:
            if (
                px - node_size / 2 < margin
                or px + node_size / 2 > width - margin
                or py - node_size / 2 < margin
                or py + node_size / 2 > content_bottom - margin
            ):
                continue
            if any(
                _boxes_overlap(
                    px,
                    py,
                    node_size,
                    node_size,
                    bx,
                    by,
                    bw,
                    bh,
                    gap=8.0,
                )
                for bx, by, bw, bh in placed_boxes
            ):
                continue

            score = hypot(px - anchor_x, py - anchor_y) * 0.30
            for ax, ay in atom_points:
                d = hypot(px - ax, py - ay)
                if d < 42:
                    score += 1400
                elif d < 68:
                    score += (68 - d) * 14

            radial_projection = (px - cx) * ux + (py - cy) * uy
            score -= radial_projection * 0.06

            for x1, y1, x2, y2 in placed_leaders:
                if _segments_intersect(
                    anchor_x,
                    anchor_y,
                    px,
                    py,
                    x1,
                    y1,
                    x2,
                    y2,
                ):
                    score += 500

            if score < best:
                best = score
                chosen = (px, py)

        if chosen is None:
            px = min(
                max(anchor_x + ux * 64, node_size / 2 + margin),
                width - node_size / 2 - margin,
            )
            py = min(
                max(anchor_y + uy * 64, node_size / 2 + margin),
                content_bottom - node_size / 2 - margin,
            )
            chosen = (px, py)

        px, py = chosen
        out[key] = {
            "x": px,
            "y": py,
            "anchor": (anchor_x, anchor_y),
            "site": (site_x, site_y),
        }
        placed_boxes.append((px, py, node_size, node_size))
        placed_leaders.append((anchor_x, anchor_y, px, py))

    return out


def _residue_positions(
    records,
    atom_by_name,
    coords,
    ligand_center,
    chemistry,
    width,
    height,
):
    """Compatibility wrapper returning only bubble centers."""
    layout = _residue_layout(
        records,
        atom_by_name,
        coords,
        ligand_center,
        chemistry,
        width,
        height,
    )
    return {key: (value["x"], value["y"]) for key, value in layout.items()}


def _outward_unit(key, x, y, center):
    cx, cy = center
    vx, vy = x - cx, y - cy
    length = hypot(vx, vy)
    if length >= 12.0:
        return vx / length, vy / length
    token = f"{key[0]}:{key[1]}:{key[2]}"
    phase = (sum(ord(ch) for ch in token) % 360) * pi / 180.0
    return cos(phase), sin(phase)


def _leader_geometry(layout):
    x, y = layout["x"], layout["y"]
    ax, ay = layout["anchor"]
    dx, dy = x - ax, y - ay
    length = hypot(dx, dy)
    if length <= 26.0:
        return None
    ux, uy = dx / length, dy / length
    radius = 30.0
    return (
        ax + ux * 7.0,
        ay + uy * 7.0,
        x - ux * (radius + 3.0),
        y - uy * (radius + 3.0),
    )


def _distance_label_layout(lines, bubble_boxes, width, content_bottom):
    placed = []
    out = []

    sortable = []
    for item in lines:
        d = _primary_distance(item["record"])
        if d is None:
            continue
        length = hypot(
            item["distance_x2"] - item["distance_x1"],
            item["distance_y2"] - item["distance_y1"],
        )
        sortable.append((length, item, d))
    sortable.sort(key=lambda x: x[0])

    for length, item, distance in sortable:
        if length < 1e-9:
            continue
        x0, y0 = item["distance_x1"], item["distance_y1"]
        x1, y1 = item["distance_x2"], item["distance_y2"]
        dx, dy = x1 - x0, y1 - y0
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx

        text = f"{distance:.2f} Å"
        fs = max(14, min(18, int(min(width, content_bottom) * 0.018)))
        tw = max(42.0, len(text) * fs * 0.56)
        th = fs * 1.05

        candidates = []
        for frac in (0.50, 0.43, 0.57, 0.35, 0.65, 0.25, 0.75):
            for off in (0, 4, -4, 8, -8, 14, -14, 20, -20):
                mx = x0 + dx * frac
                my = y0 + dy * frac
                candidates.append((mx + nx * off, my + ny * off))

        chosen = None
        best = float("inf")
        for px, py in candidates:
            if (
                px - tw / 2 < 7
                or px + tw / 2 > width - 7
                or py - th / 2 < 7
                or py + th / 2 > content_bottom - 7
            ):
                continue

            score = _point_line_distance(px, py, x0, y0, x1, y1) * 1.2
            if any(
                _boxes_overlap(px, py, tw, th, bx, by, bw, bh, gap=4.0)
                for bx, by, bw, bh in bubble_boxes
            ):
                score += 1000
            if any(
                _boxes_overlap(px, py, tw, th, bx, by, bw, bh, gap=4.0)
                for bx, by, bw, bh in placed
            ):
                score += 700

            if score < best:
                best = score
                chosen = (px, py)

        if chosen is None:
            continue
        px, py = chosen
        placed.append((px, py, tw, th))
        out.append({
            "x": px,
            "y": py,
            "w": tw,
            "h": th,
            "text": text,
        })
    return out


def _legend_layout(records, width, height):
    present = {r.interaction_type for r in records}
    ordered = [t for t in INTERACTION_ORDER if t in present]
    if not ordered:
        return None

    fs = 15 if height <= 800 else 17
    sample = 30.0
    text_gap = 8.0
    item_gap = 22.0
    max_width = max(260.0, min(float(width) - 80.0, 900.0))

    entries = []
    for interaction_type in ordered:
        label = LABELS.get(interaction_type, interaction_type)
        text_w = len(label) * fs * 0.56
        entries.append({
            "type": interaction_type,
            "label": label,
            "color": COLORS.get(interaction_type, "#595959"),
            "dash": DASH.get(interaction_type, "9 5"),
            "width": sample + text_gap + text_w,
        })

    rows = []
    current = []
    current_w = 0.0
    for entry in entries:
        extra = entry["width"] + (item_gap if current else 0.0)
        if current and current_w + extra > max_width - 30.0:
            rows.append((current, current_w))
            current = [entry]
            current_w = entry["width"]
        else:
            current.append(entry)
            current_w += extra
    if current:
        rows.append((current, current_w))

    row_h = fs + 10.0
    box_w = min(max_width, max(w for _, w in rows) + 30.0)
    box_h = len(rows) * row_h + 18.0
    return {
        "x": width / 2.0,
        "y": height - 18.0 - box_h / 2.0,
        "w": box_w,
        "h": box_h,
        "font_size": fs,
        "sample": sample,
        "text_gap": text_gap,
        "item_gap": item_gap,
        "row_h": row_h,
        "rows": rows,
    }


def _render_legend(legend):
    x, y = legend["x"], legend["y"]
    w, h = legend["w"], legend["h"]
    fs = legend["font_size"]
    parts = [
        '<g id="pv-publication-legend">',
        f'<rect class="pv-legend-box" x="{x-w/2:.1f}" y="{y-h/2:.1f}" '
        f'width="{w:.1f}" height="{h:.1f}" rx="4" '
        'fill="#FFFFFF" fill-opacity=".98" stroke="#111111" stroke-width="1.35"/>',
    ]

    top = y - h / 2.0 + 9.0
    for row_index, (items, row_width) in enumerate(legend["rows"]):
        cursor = x - row_width / 2.0
        baseline = top + row_index * legend["row_h"] + fs
        sample_y = baseline - fs * 0.38
        for entry in items:
            parts.append(
                f'<line x1="{cursor:.1f}" y1="{sample_y:.1f}" '
                f'x2="{cursor+legend["sample"]:.1f}" y2="{sample_y:.1f}" '
                f'stroke="{entry["color"]}" stroke-width="3.0" '
                f'stroke-dasharray="{entry["dash"]}" stroke-linecap="butt"/>'
            )
            cursor += legend["sample"] + legend["text_gap"]
            parts.append(
                f'<text class="pv-legend" x="{cursor:.1f}" y="{baseline:.1f}" '
                f'font-size="{fs}">{escape(entry["label"])}</text>'
            )
            cursor += len(entry["label"]) * fs * 0.56 + legend["item_gap"]
    parts.append('</g>')
    return parts


def _point_line_distance(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    qx, qy = ax + t * vx, ay + t * vy
    return hypot(px - qx, py - qy)


def _boxes_overlap(cx, cy, w, h, bx, by, bw, bh, gap=4.0):
    return not (
        cx + w / 2 + gap < bx - bw / 2
        or cx - w / 2 - gap > bx + bw / 2
        or cy + h / 2 + gap < by - bh / 2
        or cy - h / 2 - gap > by + bh / 2
    )


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


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


def _primary_distance(rec):
    for key in (
        "distance",
        "donor_acceptor_distance",
        "center_distance",
        "centroid_distance",
        "ligand_water_distance",
        "ligand_metal_distance",
    ):
        value = rec.measurements.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None
