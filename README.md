# panVizPlus

**panVizPlus** is a provenance-aware protein–ligand interaction analysis engine and Streamlit application.

## Architecture

```text
PDB / PDBQT
   ↓
structure normalization
   ↓
RDKit ligand chemistry
   ├─ bond orders
   ├─ formal charge
   ├─ donor / acceptor features
   ├─ ionizable sites
   ├─ hydrophobes
   ├─ aromatic rings
   └─ 2D coordinates
   ↓
panVizPlus interaction rules
   ↓
InteractionRecord[]
   ↓
native diagram + scientific table + CSV/JSON/SVG
```

RDKit is used as a **chemistry toolkit**, not as a protein–ligand interaction detector. Interaction classification, geometry, cutoffs, provenance, and rule versioning are implemented by panVizPlus.

## Current interaction families

- conventional hydrogen bond
- hydrophobic contact
- salt bridge
- π–π stacked
- π–π T-shaped
- π–cation
- halogen bond
- metal coordination
- water bridge
- unfavorable van der Waals contact

## Ligand chemistry

For PDB/PDBQT workflows panVizPlus builds the ligand molecular graph, uses RDKit bond-order perception and `BaseFeatures.fdef` chemical features, and asks the user for the ligand net charge used in perception. PDB input alone is not treated as authoritative for protonation or bond order.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Community Cloud:
- repository: `peterDataScientia/panVizPlus`
- branch: `main`
- entrypoint: `app.py`

Scientific defaults are versioned in `panvizplus/rules/panvizplus_v1.yaml`.

Version `0.3.0-alpha` introduces the RDKit chemistry layer while retaining panVizPlus-native interaction detection.
