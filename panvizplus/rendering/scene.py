"""PanViz presentation-scene adapter for the panVizPlus native detector.

This module deliberately keeps presentation behavior compatible with PanViz.
The only scientific substitution is the source of interaction records:
InteractionRecord objects produced by the panVizPlus engine.
"""

from __future__ import annotations

from collections import Counter
import json
from math import cos, hypot, pi, sin
from pathlib import Path
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

from panvizplus.chemistry.models import NormalizedStructure
from panvizplus.chemistry.rdkit_layer import LigandChemistry, build_ligand_chemistry
from panvizplus.interactions.models import InteractionRecord


BUBBLE_COLOR = "#0AFFEF"
NONCOVALENT_COLOR = "#595959"

TYPE_CODE = {
    "hydrophobic_contact": "HPI",
    "conventional_hbond": "HB",
    "water_bridge": "WB",
    "salt_bridge": "SB",
    "pi_pi_stacked": "PS",
    "pi_pi_t_shaped": "PS",
    "pi_cation": "PC",
    "halogen_bond": "XB",
    "metal_coordination": "MC",
    "unfavorable_vdw_bump": "UV",
}
INTERACTION_ORDER = ("HPI", "HB", "WB", "SB", "PS", "PC", "XB", "MC", "UV")
INTERACTION_COLORS = {
    "HPI": "#595959",
    "HB": "#0000E0",
    "WB": "#1596B8",
    "SB": "#D900B0",
    "PS": "#008500",
    "PC": "#E08000",
    "XB": "#7A5CC7",
    "MC": "#A45700",
    "UV": "#C62828",
}
INTERACTION_LABELS = {
    "HPI": "Hydrophobic contact",
    "HB": "Hydrogen bond",
    "WB": "Water bridge",
    "SB": "Salt bridge",
    "PS": "π-Stacking",
    "PC": "π-Cation",
    "XB": "Halogen bond",
    "MC": "Metal coordination",
    "UV": "Unfavorable contact",
}
INTERACTION_DASH = {
    "HPI": "9 5",
    "HB": "9 5",
    "WB": "3 3",
    "SB": "9 5",
    "PS": "9 5",
    "PC": "9 5",
    "XB": "6 4",
    "MC": "2 3",
    "UV": "2 3",
}
INTERACTION_SHIFT = {
    "HPI": 0.0,
    "HB": 6.0,
    "PS": 10.0,
    "PC": -10.0,
    "SB": -5.0,
    "WB": 4.0,
    "XB": -8.0,
    "MC": 8.0,
    "UV": -4.0,
}


def build_editor_scene(
    structure: NormalizedStructure,
    ligand_selector: str,
    records: list[InteractionRecord],
    width: int = 1200,
    height: int = 850,
    ligand_net_charge: int | None = None,
) -> dict:
    """Build the PanViz-compatible editable presentation scene."""
    chemistry = build_ligand_chemistry(
        structure,
        ligand_selector,
        net_charge=ligand_net_charge,
    )
    mol = Chem.Mol(chemistry.mol)
    mol.UpdatePropertyCache(strict=False)
    rdDepictor.Compute2DCoords(mol, canonOrient=True)
    conf = mol.GetConformer()

    atom_map = structure.atom_map()
    raw_atoms = []
    rd_to_name: dict[int, str] = {}
    raw_by_atom_id: dict[int, tuple[float, float]] = {}
    raw_by_name: dict[str, tuple[float, float]] = {}

    for idx, atom in enumerate(mol.GetAtoms()):
        atom_id = chemistry.rd_idx_to_atom_id.get(idx)
        if atom_id is None or atom_id not in atom_map:
            continue
        source = atom_map[atom_id]
        p = conf.GetAtomPosition(idx)
        x, y = float(p.x), float(p.y)
        rd_to_name[idx] = source.name
        raw_by_atom_id[atom_id] = (x, y)
        raw_by_name[source.name] = (x, y)
        raw_atoms.append((idx, x, y, atom.GetSymbol(), source.name))

    if not raw_atoms:
        raise ValueError("Unable to construct ligand drawing scene.")

    bonds = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in rd_to_name or b not in rd_to_name:
            continue
        bt = str(bond.GetBondType()).split(".")[-1]
        bonds.append({
            "id": f"b{a}_{b}",
            "a": a,
            "b": b,
            "order": 2 if bt in {"DOUBLE", "AROMATIC"} else 3 if bt == "TRIPLE" else 1,
            "aromatic": bt == "AROMATIC",
        })

    residue_records: dict[tuple[str, int, str], list[InteractionRecord]] = {}
    for rec in records:
        key = (rec.residue_name, rec.residue_number, rec.chain_id or "-")
        residue_records.setdefault(key, []).append(rec)

    initial_residue_raw = _raw_residue_seed_positions(
        residue_records,
        raw_by_name,
        raw_by_atom_id,
        chemistry,
    )

    all_points = [(x, y) for _, x, y, _, _ in raw_atoms]
    all_points.extend(initial_residue_raw.values())
    min_x = min(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_x = max(p[0] for p in all_points)
    max_y = max(p[1] for p in all_points)
    x_range = max(max_x - min_x, 1e-6)
    y_range = max(max_y - min_y, 1e-6)
    padding = 35.0
    usable_w = max(width - 2.0 * padding, 1.0)
    usable_h = max(height - 2.0 * padding, 1.0)
    scale = min(usable_w / x_range, usable_h / y_range)
    screen_cx, screen_cy = width / 2.0, height / 2.0
    data_cx, data_cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

    def screen_xy(x: float, y: float) -> tuple[float, float]:
        return (
            screen_cx + (x - data_cx) * scale,
            screen_cy + (y - data_cy) * scale,
        )

    atom_screen = {
        idx: screen_xy(x, y)
        for idx, x, y, _symbol, _name in raw_atoms
    }
    lig_cx = float(np.mean([p[0] for p in atom_screen.values()]))
    lig_cy = float(np.mean([p[1] for p in atom_screen.values()]))

    residue_points = [
        (*screen_xy(x, y), key)
        for key, (x, y) in initial_residue_raw.items()
    ]
    residue_font_size = max(17, min(23, int(min(width, height) * 0.025)))
    distance_font_size = max(14, min(18, int(min(width, height) * 0.018)))
    residue_draw = _place_residue_labels(
        residue_points,
        list(atom_screen.values()),
        (lig_cx, lig_cy),
        width,
        height,
        residue_font_size,
    )

    residue_id_by_key = {}
    labels = []
    for i, item in enumerate(residue_draw):
        key = item["key"]
        rid = f"res_{i}"
        residue_id_by_key[key] = rid
        resname, resnum, chain = key
        source = f"{resname}{resnum}_{chain}"
        labels.append({
            "id": rid,
            "text": f"{resname}{resnum}:{chain}",
            "sourceResidue": source,
            "x": float(item["x"]),
            "y": float(item["y"]),
            "anchorX": float(item["anchorX"]),
            "anchorY": float(item["anchorY"]),
            "w": float(item["w"]),
            "h": float(item["h"]),
            "fontSize": int(residue_font_size),
            "fontFamily": "Arial",
            "textColor": "#000000",
            "nodeShape": "bubble",
            "nodeSize": 60,
            "styleOverrides": {},
            "bold": True,
            "italic": False,
            "underline": False,
            "rotation": 0,
            "backgroundEnabled": True,
            "backgroundColor": "#ffffff",
            "backgroundOpacity": 0.98,
            "bubbleColor": BUBBLE_COLOR,
            "leaderVisible": bool(item["leader"]),
            "visible": True,
        })

    scene_atoms = []
    for idx, _x, _y, symbol, name in raw_atoms:
        px, py = atom_screen[idx]
        scene_atoms.append({
            "id": idx,
            "name": name,
            "element": symbol,
            "x": float(px),
            "y": float(py),
            "showLabel": symbol != "C",
        })

    pair_counts = Counter()
    raw_interactions = []
    for rec in records:
        code = TYPE_CODE.get(rec.interaction_type)
        if code is None:
            continue
        key = (rec.residue_name, rec.residue_number, rec.chain_id or "-")
        if key not in residue_id_by_key:
            continue
        ligand_raw, anchor_name = _record_raw_anchor(
            rec,
            raw_by_name,
            raw_by_atom_id,
            chemistry,
        )
        residue_raw = initial_residue_raw[key]
        x0, y0 = screen_xy(*ligand_raw)
        x1, y1 = screen_xy(*residue_raw)
        length = max(hypot(x1 - x0, y1 - y0), 1e-9)
        shift = INTERACTION_SHIFT[code]
        ox = (-(y1 - y0) / length) * shift
        oy = ((x1 - x0) / length) * shift
        pair_key = (key, anchor_name)
        pair_counts[pair_key] += 1
        raw_interactions.append({
            "record": rec,
            "code": code,
            "key": key,
            "anchor_name": anchor_name,
            "pair_key": pair_key,
            "x1": x0 + ox,
            "y1": y0 + oy,
            "x2": x1,
            "y2": y1,
            "distance_x1": x0,
            "distance_y1": y0,
            "distance_x2": x1,
            "distance_y2": y1,
            "distance": _primary_distance(rec),
        })

    distance_layout = _distance_positions(
        raw_interactions,
        residue_draw,
        width,
        height,
        distance_font_size,
    )

    interactions = []
    distances = []
    for row_idx, item in enumerate(raw_interactions):
        rec = item["record"]
        code = item["code"]
        iid = f"int_{row_idx}"
        multiplicity = max(1, int(pair_counts[item["pair_key"]]))
        interactions.append({
            "id": iid,
            "type": code,
            "residueId": residue_id_by_key[item["key"]],
            "sourceResidue": f"{item['key'][0]}{item['key'][1]}_{item['key'][2]}",
            "anchorAtom": item["anchor_name"],
            "x1": float(item["x1"]),
            "y1": float(item["y1"]),
            "x2": float(item["x2"]),
            "y2": float(item["y2"]),
            "originalDistance": item["distance"],
            "multiplicity": multiplicity,
            "color": INTERACTION_COLORS[code],
            "visible": True,
            "lineWidth": 3.0,
            "customLineWidth": False,
            "customColor": False,
            "customDash": False,
            "dash": INTERACTION_DASH[code],
            "rotation": 0,
            "scientificSourceId": rec.interaction_id,
        })
        dl = distance_layout.get(row_idx)
        if dl is not None and item["distance"] is not None:
            distances.append({
                "id": f"dist_{row_idx}",
                "interactionId": iid,
                "originalDistance": float(item["distance"]),
                "displayText": dl["text"],
                "x": float(dl["x"]),
                "y": float(dl["y"]),
                "w": float(dl["w"]),
                "h": float(dl["h"]),
                "fontSize": int(distance_font_size),
                "fontFamily": "Arial",
                "textColor": "#D100A0",
                "backgroundColor": "#ffffff",
                "backgroundOpacity": 0.96,
                "styleOverrides": {},
                "bold": True,
                "italic": False,
                "underline": False,
                "rotation": 0,
                "visible": True,
                "scientificSourceId": rec.interaction_id,
            })

    present = {x["type"] for x in interactions}
    legend_items = [
        {
            "type": code,
            "label": INTERACTION_LABELS[code],
            "color": INTERACTION_COLORS[code],
            "lineWidth": 3.0,
            "dash": INTERACTION_DASH[code],
            "visible": True,
        }
        for code in INTERACTION_ORDER
        if code in present
    ]
    legend_font_size = 15 if height <= 800 else 17
    legend_w = max(260.0, min(float(width) - 80.0, 900.0))
    legend = {
        "x": float(width / 2.0),
        "y": float(height - 28.0),
        "dx": 0.0,
        "dy": 0.0,
        "w": float(legend_w),
        "h": float(legend_font_size + 20),
        "fontSize": int(legend_font_size),
        "fontFamily": "Arial",
        "bold": True,
        "italic": False,
        "underline": False,
        "rotation": 0,
        "items": legend_items,
        "visible": True,
        "movable": True,
    }

    scientific = []
    for item, rendered in zip(raw_interactions, interactions):
        rec = item["record"]
        scientific.append({
            "id": rendered["id"],
            "scientificSourceId": rec.interaction_id,
            "type": rendered["type"],
            "nativeType": rec.interaction_type,
            "residueId": rendered["residueId"],
            "sourceResidue": rendered["sourceResidue"],
            "anchorAtom": rendered["anchorAtom"],
            "originalDistance": rendered["originalDistance"],
            "multiplicity": rendered["multiplicity"],
            "origin": rec.origin,
            "ruleset": rec.ruleset,
            "detector": rec.detector,
            "measurements": dict(rec.measurements),
            "metadata": dict(rec.metadata),
        })

    return {
        "version": "6.0.0-scene-compatible",
        "width": int(width),
        "height": int(height),
        "style": {
            "bondWidth": 2.0,
            "interactionWidth": 3.0,
            "noncovalentColor": NONCOVALENT_COLOR,
            "moleculeLabelSize": max(14, min(19, int(min(width, height) * 0.019))),
            "molecule": {
                "bondWidth": 2.0,
                "bondColor": "#000000",
                "colorMode": "element",
                "uniformColor": "#000000",
                "labelSize": max(14, min(19, int(min(width, height) * 0.019))),
                "labelWeight": 700,
            },
            "residue": {
                "labelFormat": "nameNumChain",
                "fontFamily": "Arial",
                "fontSize": int(residue_font_size),
                "fontWeight": 700,
                "textColor": "#000000",
                "shape": "bubble",
                "backgroundColor": "#ffffff",
                "backgroundOpacity": 0.98,
                "bubbleColor": BUBBLE_COLOR,
                "colorMode": "neutral",
                "nodeSize": 60,
            },
            "distance": {
                "fontFamily": "Arial",
                "fontSize": int(distance_font_size),
                "fontWeight": 700,
                "textColor": "#D100A0",
                "backgroundColor": "#ffffff",
                "backgroundOpacity": 0.96,
            },
            "interactions": {
                code: {
                    "color": INTERACTION_COLORS[code],
                    "width": 3.0,
                    "dash": INTERACTION_DASH[code],
                    "custom": False,
                }
                for code in INTERACTION_ORDER
            },
        },
        "site": ligand_selector,
        "atoms": scene_atoms,
        "bonds": bonds,
        "labels": labels,
        "interactions": interactions,
        "distances": distances,
        "legend": legend,
        "notes": [],
        "arrows": [],
        "customInteractions": [],
        "bondEdits": [],
        "metadata": {
            "editorMode": "layer-separated-scene",
            "chemistryEditable": False,
            "bondGraphicsEditable": True,
            "annotationLayerFullyEditable": True,
            "scientificDataImmutable": True,
            "initialView": "panviz-layout-scene",
            "canvasResize": "proportional-content-scale-fit",
            "canvasScrollbars": False,
            "contextualRibbon": "external-overlay",
            "ribbonFrozenDuringDrag": True,
            "separateInteractionWidthControl": True,
            "semanticResidueNodeStyles": ["plain", "circle", "plate", "rounded", "bubble"],
            "globalLocalStyleHierarchy": True,
            "globalDefaults": {
                "covalentWidth": 2.0,
                "noncovalentWidth": 3.0,
                "bubbleColor": BUBBLE_COLOR,
            },
            "duplicateInteractionWeight": "baseWidth × interactionMultiplicity",
            "smartResidueLeaderAttachment": True,
            "moleculeLocked": True,
            "annotationOnly": True,
            "ghostFree": True,
            "detector": "panVizPlus-native",
            "presentationCompatibility": "PanViz",
        },
        "scientificData": {
            "interactions": scientific,
            "detector": "panVizPlus-native",
        },
    }


def render_editor_html(scene: dict) -> str:
    """Inject a native scene into the PanViz-compatible editor."""
    template = Path(__file__).with_name("editor.html").read_text(encoding="utf-8")
    html = template.replace("__PANVIZ_SCENE__", json.dumps(scene, ensure_ascii=False))
    return html + """
<script>
(() => {
  let lastHeight = 0;
  const reportHeight = () => {
    const root = document.getElementById('pv-root');
    if (!root) return;
    const height = Math.ceil(Math.max(
      root.scrollHeight,
      root.getBoundingClientRect().height,
      document.body?.scrollHeight || 0,
      document.documentElement?.scrollHeight || 0
    ) + 8);
    if (Math.abs(height - lastHeight) < 2) return;
    lastHeight = height;
    window.parent.postMessage({
      isStreamlitMessage: true,
      type: 'streamlit:setFrameHeight',
      height
    }, '*');
  };
  window.addEventListener('load', () => {
    reportHeight();
    setTimeout(reportHeight, 80);
    setTimeout(reportHeight, 350);
  });
  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(reportHeight);
    observer.observe(document.documentElement);
    const root = document.getElementById('pv-root');
    if (root) observer.observe(root);
  }
})();
</script>
"""


def _record_raw_anchor(
    rec: InteractionRecord,
    raw_by_name: dict[str, tuple[float, float]],
    raw_by_atom_id: dict[int, tuple[float, float]],
    chemistry: LigandChemistry,
) -> tuple[tuple[float, float], str]:
    site = str(rec.ligand_site)
    name = site.split(":")[-1]
    if name in raw_by_name:
        return raw_by_name[name], name

    ring_match = re.search(r"aromatic_ring(\d+)", site)
    if ring_match:
        ring_index = int(ring_match.group(1)) - 1
        if 0 <= ring_index < len(chemistry.aromatic_rings):
            ids = chemistry.aromatic_rings[ring_index]
            pts = [raw_by_atom_id[atom_id] for atom_id in ids if atom_id in raw_by_atom_id]
            if pts:
                return (
                    (
                        float(np.mean([p[0] for p in pts])),
                        float(np.mean([p[1] for p in pts])),
                    ),
                    f"ring{ring_index + 1}",
                )

    pts = list(raw_by_atom_id.values())
    return (
        (
            float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])),
        ),
        "ligand",
    )


def _raw_residue_seed_positions(grouped, raw_by_name, raw_by_atom_id, chemistry):
    occupied = np.asarray(list(raw_by_atom_id.values()), dtype=float)
    placed: list[tuple[float, float]] = []
    out = {}

    for key, recs in grouped.items():
        anchors = [
            _record_raw_anchor(rec, raw_by_name, raw_by_atom_id, chemistry)[0]
            for rec in recs
        ]
        initial = np.asarray([
            float(np.mean([p[0] for p in anchors])),
            float(np.mean([p[1] for p in anchors])),
        ], dtype=float)

        offsets = np.mgrid[-3.5:3.6:0.2, -3.5:3.6:0.2].reshape(2, -1).T
        candidates = offsets + initial
        best = None
        best_score = float("inf")

        for cand in candidates:
            score = 0.0
            if occupied.size:
                dx = np.abs(occupied[:, 0] - cand[0])
                dy = np.abs(occupied[:, 1] - cand[1])
                score += float(np.sum((dx < 2.0) & (dy < 2.0)))
            for px, py in placed:
                if abs(cand[0] - px) < 2.5 and abs(cand[1] - py) < 2.0:
                    score += 1.0
            score += hypot(float(cand[0] - initial[0]), float(cand[1] - initial[1])) * 0.001
            if score < best_score:
                best_score = score
                best = cand
                if score <= 0.001:
                    break

        if best is None:
            best = initial
        pos = (float(best[0]), float(best[1]))
        out[key] = pos
        placed.append(pos)
    return out


def _place_residue_labels(residue_points, atom_screen, ligand_center, width, height, font_size):
    lig_cx, lig_cy = ligand_center
    placed_boxes = []
    out = []

    residue_points = sorted(
        residue_points,
        key=lambda p: hypot(p[0] - lig_cx, p[1] - lig_cy),
    )
    for anchor_x, anchor_y, key in residue_points:
        resname, resnum, chain = key
        display = f"{resname}{resnum}:{chain}"
        tw = max(42.0, len(display) * font_size * 0.56)
        th = max(18.0, font_size * 1.06)

        vx, vy = anchor_x - lig_cx, anchor_y - lig_cy
        vlen = hypot(vx, vy)
        if vlen < 1e-6:
            phase = (sum(ord(c) for c in display) % 360) * pi / 180.0
            ux, uy = cos(phase), sin(phase)
        else:
            ux, uy = vx / vlen, vy / vlen
        nx, ny = -uy, ux

        candidates = []
        for radial in (20, 28, 38, 50, 64, 80, 98):
            for tangent in (0, 8, -8, 16, -16):
                candidates.append((
                    anchor_x + ux * radial + nx * tangent,
                    anchor_y + uy * radial + ny * tangent,
                ))
        for radial in (24, 36, 52):
            candidates.append((anchor_x - ux * radial, anchor_y - uy * radial))

        chosen = None
        best_score = float("inf")
        for px, py in candidates:
            bw = max(60.0, tw)
            bh = max(60.0, th)
            if (
                px - bw / 2 < 10
                or px + bw / 2 > width - 10
                or py - bh / 2 < 10
                or py + bh / 2 > height - 10
            ):
                continue

            score = hypot(px - anchor_x, py - anchor_y) * 0.30
            for ax, ay in atom_screen:
                d = hypot(px - ax, py - ay)
                if d < 32:
                    score += 1200
                elif d < 56:
                    score += (56 - d) * 12

            if any(
                _boxes_overlap(px, py, bw, bh, bx, by, pbw, pbh, gap=8)
                for bx, by, pbw, pbh in placed_boxes
            ):
                continue

            radial_projection = (px - lig_cx) * ux + (py - lig_cy) * uy
            score -= radial_projection * 0.06
            if score < best_score:
                best_score = score
                chosen = (px, py)

        if chosen is None:
            chosen = (
                max(40.0, min(width - 40.0, anchor_x)),
                max(40.0, min(height - 40.0, anchor_y)),
            )

        px, py = chosen
        placed_boxes.append((px, py, max(60.0, tw), max(60.0, th)))
        out.append({
            "key": key,
            "x": px,
            "y": py,
            "anchorX": anchor_x,
            "anchorY": anchor_y,
            "w": tw,
            "h": th,
            "leader": hypot(px - anchor_x, py - anchor_y) > 26.0,
        })
    return out


def _distance_positions(interaction_lines, residue_draw, width, height, font_size):
    residue_boxes = [
        (item["x"], item["y"], max(60.0, item["w"]), max(60.0, item["h"]))
        for item in residue_draw
    ]
    placed = []
    result = {}

    ranked = []
    for idx, item in enumerate(interaction_lines):
        if item["distance"] is None:
            continue
        length = hypot(
            item["distance_x2"] - item["distance_x1"],
            item["distance_y2"] - item["distance_y1"],
        )
        ranked.append((length, idx, item))
    ranked.sort(key=lambda x: x[0])

    for length, idx, item in ranked:
        if length < 1e-9:
            continue
        x0, y0 = item["distance_x1"], item["distance_y1"]
        x1, y1 = item["distance_x2"], item["distance_y2"]
        dx, dy = x1 - x0, y1 - y0
        nx, ny = -dy / length, dx / length
        text = f"{float(item['distance']):.2f} Å"
        tw = max(42.0, len(text) * font_size * 0.56)
        th = max(14.0, font_size * 1.06)

        candidates = []
        for frac in (0.50, 0.43, 0.57, 0.35, 0.65, 0.25, 0.75):
            for off in (0, 4, -4, 8, -8, 14, -14, 20, -20):
                mx, my = x0 + dx * frac, y0 + dy * frac
                candidates.append((mx + nx * off, my + ny * off))

        chosen = None
        best_score = float("inf")
        for px, py in candidates:
            if (
                px - tw / 2 < 7
                or px + tw / 2 > width - 7
                or py - th / 2 < 7
                or py + th / 2 > height - 7
            ):
                continue
            score = _point_line_distance(px, py, x0, y0, x1, y1) * 1.2
            if any(
                _boxes_overlap(px, py, tw, th, bx, by, bw, bh, gap=4)
                for bx, by, bw, bh in residue_boxes
            ):
                score += 1000
            if any(
                _boxes_overlap(px, py, tw, th, bx, by, bw, bh, gap=4)
                for bx, by, bw, bh in placed
            ):
                score += 700
            if score < best_score:
                best_score = score
                chosen = (px, py)

        if chosen is None:
            continue
        px, py = chosen
        placed.append((px, py, tw, th))
        result[idx] = {"x": px, "y": py, "w": tw, "h": th, "text": text}
    return result


def _primary_distance(rec: InteractionRecord) -> float | None:
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
