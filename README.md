# panVizPlus

**panVizPlus** is a native, provenance-aware protein–ligand interaction analysis engine and Streamlit application.

## Scientific architecture

```text
PDB / PDBQT
   ↓
structure normalization
   ↓
chemical feature perception
   ↓
panVizPlus interaction engine
   ↓
versioned geometric rules
   ↓
InteractionRecord[]
   ↓
native 2D diagram + table + CSV/JSON/SVG
```

The interaction engine is implemented inside `panvizplus/interactions/`. It does not call an external protein–ligand interaction detector.

## Current native interaction families

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

Every interaction record stores ligand/protein sites, residue identity, measurements, criteria, detector version, ruleset version, chemistry-confidence metadata, and provenance.

## Chemistry policy

PDB-only ligand chemistry can be incomplete. panVizPlus therefore marks connectivity-derived ligand chemical features with lower confidence instead of silently treating them as authoritative. PDBQT files are converted to PDB for coordinate/connectivity normalization; Open Babel is used only as a file-format converter.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

- repository: `peterDataScientia/panVizPlus`
- branch: `main`
- entrypoint: `app.py`

## Rule registry

Scientific defaults are stored in:

`panvizplus/rules/panvizplus_v1.yaml`

Version `0.2.0-alpha` is an alpha scientific implementation. Rule thresholds are explicit and benchmarkable and are not presented as exact reproductions of proprietary software.
