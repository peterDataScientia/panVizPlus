# panVizPlus

**panVizPlus** is an open, provenance-aware framework for scientifically defensible protein–ligand interaction analysis and publication-quality 2D visualization.

> **Project status:** early scientific-engineering foundation. The interaction rules and chemistry layer are being defined before the visualization UI is expanded.

## Design principle

panVizPlus separates four concerns that must not be mixed:

1. **Chemistry normalization** — ligand identity, connectivity, bond order, aromaticity, formal charge, protonation metadata, waters/ions/cofactors.
2. **Interaction perception** — chemically typed, interaction-specific geometric rules.
3. **Provenance model** — every detected or manually added interaction retains atoms/sites, residue identity, measured geometry, rule version, detector, and source.
4. **2D rendering and curation** — visualization consumes validated interaction records; the renderer does not decide chemistry.

```text
PDB / mmCIF / SDF / MOL2
          ↓
Chemistry normalization
          ↓
Chemical feature perception
          ↓
Versioned interaction rules
          ↓
Provenance-rich interaction records
          ↓
2D layout / interactive curation
          ↓
SVG / PNG / PDF + interaction table
```

## Scientific position

The project is informed by established tools including BIOVIA Discovery Studio and PLIP, but **panVizPlus does not claim to reproduce proprietary BIOVIA algorithms exactly**. Any Discovery-Studio-compatible rules are stored explicitly with provenance and confidence so they can be inspected, tested, and revised.

Modern BIOVIA releases emphasize correct chemical representation (especially ligand bond orders/formal charges and mmCIF/CCD handling) before interaction analysis. panVizPlus follows the same general scientific principle while keeping its rules open and auditable.

## Repository layout

```text
panvizplus/
  chemistry/       # structure and chemical normalization
  interactions/    # detector interfaces and interaction records
  rules/           # versioned machine-readable criteria
  layout/          # ligand/residue 2D layout
  rendering/       # SVG/PNG/PDF output
  validation/      # cross-tool and benchmark comparisons

docs/
  INTERACTION_SPECIFICATION.md

tests/
```

## Immediate roadmap

- [x] Establish package and interaction-specification architecture
- [x] Add versioned rule registry with evidence/confidence metadata
- [ ] Implement CCD-aware ligand chemistry normalization
- [ ] Implement conventional and carbon H-bonds
- [ ] Implement electrostatic and aromatic interactions
- [ ] Implement hydrophobic, halogen, sulfur, metal and water-mediated interactions
- [ ] Build benchmark suite against curated complexes and independent tools
- [ ] Integrate the proven PANVIZ editor/renderer components
- [ ] Add Streamlit application layer

## Reproducibility rule

A reported interaction must be reproducible from:

- input structure identity,
- normalized chemical state,
- detector/rule-set version,
- atom or site identities,
- measured geometric quantities,
- pass/fail criteria,
- automatic vs manual provenance.

See [docs/INTERACTION_SPECIFICATION.md](docs/INTERACTION_SPECIFICATION.md).

## Name

Project name: **panVizPlus**.
