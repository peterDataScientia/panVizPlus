from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from panviz_version import PANVIZ_VERSION
from panvizplus.chemistry.pdb import read_pdb
from panvizplus.chemistry.rdkit_layer import build_ligand_chemistry
from panvizplus.interactions.engine import analyze_structure
from panvizplus.rendering import render_interaction_svg
from panvizplus.rules import load_ruleset


st.set_page_config(page_title=f"panVizPlus {PANVIZ_VERSION}", page_icon="🧬", layout="wide")
st.markdown(
    """
<style>
:root{--navy:#143761;--ink:#16243a;--muted:#68788d;--line:#dce4ef}
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1480px}
.hero{border:1px solid #dbe3ee;border-radius:18px;padding:18px 20px;background:linear-gradient(135deg,#f7faff,#fff 55%,#f4f7fb);box-shadow:0 8px 28px rgba(24,54,90,.07);margin-bottom:14px}
.title{font-size:2rem;font-weight:800;color:var(--ink);letter-spacing:-.5px}
.sub{color:var(--muted);font-size:.95rem;margin-top:4px}
.badges{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}
.badge{font-size:.72rem;font-weight:700;color:#34506f;background:#eef4fb;border:1px solid #dce7f3;border-radius:999px;padding:5px 9px}
.card{border:1px solid var(--line);border-radius:14px;padding:12px 14px;background:#fff;box-shadow:0 4px 14px rgba(31,55,88,.05);margin:10px 0}
.card h4{margin:0 0 8px;color:var(--navy);font-size:13px}
.foot{color:#7a8798;font-size:.74rem;margin-top:10px}
</style>
<div class="hero">
  <div class="title">panVizPlus</div>
  <div class="sub">Native protein–ligand interaction analysis with explicit, versioned scientific rules.</div>
  <div class="badges">
    <span class="badge">panVizPlus-native engine</span>
    <span class="badge">RDKit chemistry</span>
    <span class="badge">panVizPlus interaction rules</span>
    <span class="badge">Rule provenance</span>
    <span class="badge">PDB + PDBQT</span>
    <span class="badge">v""" + PANVIZ_VERSION + """</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


def _workspace() -> Path:
    if "pv_native_root" not in st.session_state or not Path(st.session_state.pv_native_root).exists():
        st.session_state.pv_native_root = tempfile.mkdtemp(prefix="panvizplus_")
    return Path(st.session_state.pv_native_root)


def _find_obabel() -> str | None:
    return shutil.which("obabel") or shutil.which("obabel.exe")


def _convert_to_pdb(source: Path, destination: Path, fmt: str) -> None:
    obabel = _find_obabel()
    if not obabel:
        raise RuntimeError("Open Babel executable was not found; PDB input remains available.")
    run = subprocess.run(
        [obabel, "-i", fmt, str(source), "-o", "pdb", "-O", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0 or not destination.exists():
        raise RuntimeError((run.stderr or run.stdout or "Open Babel conversion failed.").strip())


def _pdbqt_models(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(r"MODEL\s+\d+.*?ENDMDL", text, flags=re.S | re.I)
    return blocks or [text]


def _vina_scores(blocks: list[str]) -> list[float | None]:
    out = []
    for block in blocks:
        m = re.search(r"REMARK\s+VINA\s+RESULT:\s+(-?\d+(?:\.\d+)?)", block, flags=re.I)
        out.append(float(m.group(1)) if m else None)
    return out


def _normalize_ligand_pdb(raw_pdb: Path, out_pdb: Path, start_serial: int) -> None:
    lines = raw_pdb.read_text(encoding="utf-8", errors="replace").splitlines()
    atoms, conect, serial_map = [], [], {}
    serial = start_serial
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            s = line.ljust(80)
            try:
                old = int(s[6:11])
            except ValueError:
                old = None
            if old is not None:
                serial_map[old] = serial
            atom_name = s[12:16]
            element = s[76:78].strip()
            if not element:
                raw = re.sub(r"[^A-Za-z]", "", atom_name)
                element = raw[:2].title() if raw[:2].lower() in {"cl", "br"} else raw[:1].upper()
            charge = s[78:80] if len(s) >= 80 else "  "
            atoms.append(
                f"HETATM{serial:5d} {atom_name:>4} LIG Z 900    "
                f"{s[30:38]:>8}{s[38:46]:>8}{s[46:54]:>8}  1.00  0.00          {element:>2}{charge:>2}"
            )
            serial += 1
        elif line.startswith("CONECT"):
            nums = []
            for tok in line.split()[1:]:
                try:
                    mapped = serial_map.get(int(tok))
                except ValueError:
                    mapped = None
                if mapped is not None:
                    nums.append(mapped)
            if len(nums) >= 2:
                conect.append("CONECT" + "".join(f"{n:5d}" for n in nums))
    if not atoms:
        raise RuntimeError("Converted ligand contains no atoms.")
    out_pdb.write_text("\n".join(atoms + conect + ["TER", "END"]) + "\n", encoding="utf-8")


def _prepare_docking_complex(receptor: Path, ligand: Path, pose_index: int, root: Path) -> Path:
    receptor_pdb = root / "native_receptor.pdb"
    if receptor.suffix.lower() == ".pdb":
        receptor_pdb.write_bytes(receptor.read_bytes())
    else:
        _convert_to_pdb(receptor, receptor_pdb, "pdbqt")

    blocks = _pdbqt_models(ligand)
    if pose_index >= len(blocks):
        raise RuntimeError("Selected pose is outside the available pose range.")
    pose_src = root / "selected_pose.pdbqt"
    pose_src.write_text(blocks[pose_index] + "\n", encoding="utf-8")
    raw_ligand = root / "selected_pose_raw.pdb"
    _convert_to_pdb(pose_src, raw_ligand, "pdbqt")

    receptor_lines = receptor_pdb.read_text(encoding="utf-8", errors="replace").splitlines()
    serials = []
    kept_receptor = []
    for line in receptor_lines:
        if line.startswith(("ATOM", "HETATM", "TER")):
            kept_receptor.append(line)
        if line.startswith(("ATOM", "HETATM")):
            try:
                serials.append(int(line[6:11]))
            except ValueError:
                pass

    normalized_ligand = root / "selected_pose_ligand.pdb"
    _normalize_ligand_pdb(raw_ligand, normalized_ligand, max(serials, default=0) + 1)
    lig_lines = [
        x for x in normalized_ligand.read_text(encoding="utf-8").splitlines()
        if x.startswith(("HETATM", "CONECT", "TER"))
    ]

    combined = root / "panvizplus_native_complex.pdb"
    combined.write_text("\n".join(kept_receptor + ["TER"] + lig_lines + ["END"]) + "\n", encoding="utf-8")
    return combined


def _records_frame(records) -> pd.DataFrame:
    rows = []
    for r in records:
        distance = None
        for key in (
            "distance", "donor_acceptor_distance", "center_distance", "centroid_distance",
            "ligand_water_distance", "ligand_metal_distance",
        ):
            value = r.measurements.get(key)
            if isinstance(value, (float, int)):
                distance = value
                break
        rows.append({
            "Interaction": r.interaction_type,
            "Residue": f"{r.residue_name}{r.residue_number}:{r.chain_id or '-'}",
            "Ligand site": r.ligand_site,
            "Protein site": r.protein_site,
            "Distance (Å)": distance,
            "Measurements": json.dumps(r.measurements, ensure_ascii=False),
            "Criteria": "; ".join(
                f"{c.name} {c.comparator} {c.threshold}{' '+c.units if c.units else ''}"
                for c in r.criteria
            ),
            "Chemistry confidence": r.metadata.get("chemistry_confidence") or r.metadata.get("ligand_feature_confidence"),
            "Chemistry source": r.metadata.get("chemistry_source") or r.metadata.get("ligand_feature_source"),
            "Ruleset": r.ruleset,
            "Detector": r.detector,
        })
    return pd.DataFrame(rows)


def _custom_rules() -> dict:
    rules = copy.deepcopy(load_ruleset())
    with st.expander("Advanced scientific criteria", expanded=False):
        st.caption("Defaults are versioned in the panVizPlus rule registry. Any change applies only to this analysis.")
        rules["interactions"]["conventional_hbond"]["geometry"]["donor_acceptor_distance_max"] = st.number_input(
            "H-bond D···A maximum (Å)", 2.5, 4.5,
            float(rules["interactions"]["conventional_hbond"]["geometry"]["donor_acceptor_distance_max"]), 0.1
        )
        rules["interactions"]["hydrophobic_contact"]["geometry"]["atom_distance_max"] = st.number_input(
            "Hydrophobic atom distance maximum (Å)", 3.5, 6.0,
            float(rules["interactions"]["hydrophobic_contact"]["geometry"]["atom_distance_max"]), 0.1
        )
        rules["interactions"]["salt_bridge"]["geometry"]["charge_center_distance_max"] = st.number_input(
            "Salt-bridge distance maximum (Å)", 3.0, 6.0,
            float(rules["interactions"]["salt_bridge"]["geometry"]["charge_center_distance_max"]), 0.1
        )
        rules["interactions"]["pi_pi"]["common_geometry"]["center_distance_max"] = st.number_input(
            "π–π centroid distance maximum (Å)", 4.0, 7.0,
            float(rules["interactions"]["pi_pi"]["common_geometry"]["center_distance_max"]), 0.1
        )
        rules["interactions"]["pi_cation"]["geometry"]["center_distance_max"] = st.number_input(
            "π–cation center distance maximum (Å)", 3.5, 7.0,
            float(rules["interactions"]["pi_cation"]["geometry"]["center_distance_max"]), 0.1
        )
        rules["interactions"]["halogen_bond"]["geometry"]["C_X_A_angle_min"] = st.number_input(
            "Halogen C–X···A minimum angle (°)", 100.0, 180.0,
            float(rules["interactions"]["halogen_bond"]["geometry"]["C_X_A_angle_min"]), 5.0
        )
        rules["interactions"]["metal_coordination"]["geometry"]["distance_max"] = st.number_input(
            "Metal coordination distance maximum (Å)", 2.0, 4.0,
            float(rules["interactions"]["metal_coordination"]["geometry"]["distance_max"]), 0.1
        )
        rules["interactions"]["water_bridge"]["geometry"]["heavy_atom_distance_max"] = st.number_input(
            "Water-bridge heavy-atom maximum (Å)", 2.5, 4.5,
            float(rules["interactions"]["water_bridge"]["geometry"]["heavy_atom_distance_max"]), 0.1
        )
    return rules


root = _workspace()
st.markdown('<div class="card"><h4>1 · Input structure</h4>', unsafe_allow_html=True)
mode = st.radio("Input mode", ["PDB complex", "Docking PDBQT"], horizontal=True)

source_pdb = None
source_signature = ""
pose_score = None

if mode == "PDB complex":
    uploaded = st.file_uploader("Upload protein–ligand PDB complex", type=["pdb"])
    if uploaded is not None:
        source_pdb = root / uploaded.name
        data = uploaded.getvalue()
        source_pdb.write_bytes(data)
        source_signature = hashlib.sha256(data).hexdigest()
else:
    left, right = st.columns(2)
    with left:
        receptor_upload = st.file_uploader("Protein / receptor", type=["pdb", "pdbqt"], key="native_receptor")
    with right:
        ligand_upload = st.file_uploader("Ligand / docking poses", type=["pdbqt"], key="native_ligand")
    if receptor_upload is not None and ligand_upload is not None:
        receptor = root / receptor_upload.name
        ligand = root / ligand_upload.name
        receptor.write_bytes(receptor_upload.getvalue())
        ligand.write_bytes(ligand_upload.getvalue())
        blocks = _pdbqt_models(ligand)
        scores = _vina_scores(blocks)
        pose = st.selectbox(
            "Docking pose",
            list(range(len(blocks))),
            format_func=lambda i: f"Pose {i+1}" + (f" · Vina {scores[i]:.2f} kcal/mol" if scores[i] is not None else ""),
        )
        pose_score = scores[pose]
        try:
            source_pdb = _prepare_docking_complex(receptor, ligand, pose, root)
            source_signature = hashlib.sha256(
                receptor_upload.getvalue() + ligand_upload.getvalue() + str(pose).encode()
            ).hexdigest()
        except Exception as exc:
            st.error(f"PDBQT preparation failed: {exc}")
st.markdown("</div>", unsafe_allow_html=True)

if source_pdb is None:
    st.info("Upload the required structure file(s) to begin.")
    st.stop()

try:
    structure = read_pdb(source_pdb)
except Exception as exc:
    st.error(f"Structure parsing failed: {exc}")
    st.stop()

ligand_residues = structure.ligand_residues()
if not ligand_residues:
    st.error("No non-water small-molecule HETATM residue was found.")
    st.stop()

st.markdown('<div class="card"><h4>2 · Analysis setup</h4>', unsafe_allow_html=True)
ligand_tuple = st.selectbox(
    "Ligand",
    ligand_residues,
    format_func=lambda x: f"{x[0]}:{x[1] or '-'}:{x[2]}",
)
ligand_selector = f"{ligand_tuple[0]}:{ligand_tuple[1] or '-'}:{ligand_tuple[2]}"
selected_ligand_atoms = structure.ligand_atoms(ligand_selector)
default_charge = int(sum(a.formal_charge or 0 for a in selected_ligand_atoms))
c1, c2, c3 = st.columns(3)
with c1:
    ligand_net_charge = st.number_input(
        "Ligand net charge",
        min_value=-8,
        max_value=8,
        value=default_charge,
        step=1,
        help="Used by RDKit bond-order/formal-charge perception. Set this to the ligand protonation state used for docking/MD.",
    )
with c2:
    fig_width = st.number_input("Figure width", 800, 2200, 1200, 100)
with c3:
    fig_height = st.number_input("Figure height", 600, 1600, 820, 50)
rules = _custom_rules()
run_analysis = st.button("Analyze with panVizPlus", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

rule_signature = hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()
result_key = (
    source_signature,
    ligand_selector,
    int(ligand_net_charge),
    rule_signature,
    int(fig_width),
    int(fig_height),
)

if run_analysis:
    with st.spinner("Running native panVizPlus interaction engine…"):
        chemistry = build_ligand_chemistry(
            structure,
            ligand_selector,
            net_charge=int(ligand_net_charge),
        )
        records = analyze_structure(
            structure,
            ligand_selector,
            rules,
            ligand_net_charge=int(ligand_net_charge),
        )
        frame = _records_frame(records)
        svg = render_interaction_svg(
            structure,
            ligand_selector,
            records,
            int(fig_width),
            int(fig_height),
            ligand_net_charge=int(ligand_net_charge),
        )
        st.session_state.native_result = {
            "key": result_key,
            "records": records,
            "frame": frame,
            "svg": svg,
            "warnings": list(structure.warnings) + list(chemistry.warnings),
            "rules": rules,
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
                "net_charge": int(ligand_net_charge),
            },
            "pose_score": pose_score,
        }

result = st.session_state.get("native_result")
if not result or result.get("key") != result_key:
    st.info("Click **Analyze with panVizPlus** to run the native interaction engine.")
    st.stop()

frame = result["frame"]
records = result["records"]
types = frame["Interaction"].nunique() if not frame.empty else 0
residues = frame["Residue"].nunique() if not frame.empty else 0
unfavorable = int((frame["Interaction"] == "unfavorable_vdw_bump").sum()) if not frame.empty else 0

st.markdown('<div class="card"><h4>3 · Native scientific summary</h4>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Interactions", len(records))
m2.metric("Interaction classes", int(types))
m3.metric("Interacting residues", int(residues))
m4.metric("RDKit aromatic rings", int(result.get("chemistry", {}).get("aromatic_rings", 0)))
if result.get("pose_score") is not None:
    st.caption(f"Selected docking pose Vina score: {result['pose_score']:.2f} kcal/mol")
for warning in result["warnings"]:
    st.warning(warning)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card"><h4>4 · Native interaction diagram</h4>', unsafe_allow_html=True)
components.html(result["svg"], height=int(fig_height) + 20, scrolling=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card"><h4>5 · Scientific records</h4>', unsafe_allow_html=True)
if frame.empty:
    chemistry_mode = result.get("chemistry", {}).get("reconstruction_mode")
    if chemistry_mode == "authoritative_chemistry_required":
        st.error(
            "Interaction analysis is incomplete because reliable ligand chemistry could not be "
            "resolved. This is not evidence that the complex has no interactions. For named PDB "
            "components panVizPlus attempts automatic wwPDB CCD chemistry; verify network access, "
            "component identity, and ligand charge if CCD resolution fails."
        )
    else:
        st.info("No interactions passed the currently selected panVizPlus rules.")
else:
    st.dataframe(frame, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Methodology and rule provenance", expanded=False):
    st.write(
        "RDKit supplies ligand chemistry perception (bond orders, aromaticity, formal charge, "
        "donor/acceptor and related chemical features). panVizPlus applies the protein–ligand "
        "interaction geometry and classification rules. RDKit is not used as an interaction detector."
    )
    st.subheader("Ligand chemistry")
    st.json(result.get("chemistry", {}))
    st.subheader("Interaction rules")
    st.json(result["rules"])

csv_bytes = frame.to_csv(index=False).encode("utf-8")
json_bytes = json.dumps(
    [
        {
            "interaction_id": r.interaction_id,
            "interaction_type": r.interaction_type,
            "ligand_site": r.ligand_site,
            "protein_site": r.protein_site,
            "residue": f"{r.residue_name}{r.residue_number}:{r.chain_id or '-'}",
            "measurements": r.measurements,
            "criteria": [c.__dict__ if hasattr(c, "__dict__") else {
                "name": c.name,
                "measured_value": c.measured_value,
                "comparator": c.comparator,
                "threshold": c.threshold,
                "units": c.units,
                "passed": c.passed,
            } for c in r.criteria],
            "detector": r.detector,
            "detector_version": r.detector_version,
            "ruleset": r.ruleset,
            "metadata": r.metadata,
        }
        for r in records
    ],
    indent=2,
    ensure_ascii=False,
).encode("utf-8")

d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("Download CSV", csv_bytes, "panVizPlus_interactions.csv", "text/csv", use_container_width=True)
with d2:
    st.download_button("Download JSON", json_bytes, "panVizPlus_interactions.json", "application/json", use_container_width=True)
with d3:
    st.download_button("Download SVG", result["svg"].encode("utf-8"), "panVizPlus_interaction_diagram.svg", "image/svg+xml", use_container_width=True)

st.markdown(f'<div class="foot">panVizPlus {PANVIZ_VERSION} · native interaction engine</div>', unsafe_allow_html=True)
