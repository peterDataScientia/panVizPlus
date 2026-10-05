# panVizPlus Interaction Specification

## 1. Purpose

This document defines the scientific contract between chemistry normalization,
interaction detection, provenance storage, and 2D rendering.

The central rule is:

> **The renderer must never decide whether an interaction exists.**

Detection occurs upstream and produces an auditable `InteractionRecord`.

## 2. Scope and scientific position

panVizPlus is informed by established protein-ligand interaction tools, including
BIOVIA Discovery Studio and PLIP, but it does **not** claim to reproduce any
proprietary implementation exactly.

The initial `panvizplus_v1` registry is deliberately marked **draft**. A numeric
criterion is not promoted to a stable default until its source and benchmark
behavior are documented.

## 3. Pipeline

```text
structure
  ↓
ligand/receptor/water/ion/cofactor identification
  ↓
bond-order + formal-charge + aromaticity normalization
  ↓
chemical feature perception
  ↓
candidate site generation
  ↓
interaction-specific geometry
  ↓
InteractionRecord + CriterionResult[]
  ↓
2D layout and rendering
```

## 4. Chemistry normalization requirements

Before interaction detection, panVizPlus should preserve or reconstruct, where
supported by evidence:

- atom identity and coordinates;
- covalent connectivity;
- bond orders;
- aromaticity;
- formal charges;
- ligand identity;
- protein residue/chain identity;
- water and ion identity;
- explicit hydrogens when present;
- provenance for any inferred chemistry.

For PDB/mmCIF structures, authoritative Chemical Component Dictionary (CCD)
information should be preferred over blind distance-based bond-order inference
when available.

## 5. Interaction evidence model

Every automatic interaction should record:

- interaction type(s);
- ligand atom, group, or ring identity;
- protein atom, group, or ring identity;
- residue name, number, and chain;
- measured distances and angles;
- each criterion and pass/fail result;
- detector and detector version;
- rule-set identifier/version;
- source/provenance;
- automatic vs manual origin.

A single site pair may legitimately satisfy more than one interaction type.

## 6. Hydrogen-bond geometry

The Discovery-Studio-style architecture recovered from documented API concepts
distinguishes geometry according to whether a hydrogen is explicitly present.

With explicit hydrogen:

```text
X—D—H···A—Y
    \___/
     DHA
```

Relevant geometry can include D-H-A (DHA) and H-A-Y (HAY).

Without explicit hydrogen:

```text
X—D···A—Y
 \___/\_/
  XDA  DAY
```

Heavy-atom surrogate geometry such as X-D-A and D-A-Y can be used rather than
silently inventing a hydrogen coordinate.

## 7. Draft interaction registry

The machine-readable source of truth is:

`panvizplus/rules/panvizplus_v1.yaml`

Current entries are provisional and include conventional H-bond, carbon H-bond,
attractive charge, pi-cation, pi-pi, alkyl, pi-alkyl, and unfavorable van der
Waals bump rules.

### Important limitation

The values are **DS-compatible research baselines**, not a claim of exact modern
BIOVIA Discovery Studio 2026 defaults. Modern BIOVIA release notes document major
improvements in chemical representation, particularly ligand bond order/formal
charge and mmCIF/CCD handling, but do not publicly enumerate every modern
`Mdm::NonbondCriterionType` default.

## 8. Rule promotion policy

A rule can move from `provisional` to `stable` only after:

1. chemical prerequisites are explicitly defined;
2. geometry and units are explicit;
3. source provenance is recorded;
4. edge cases are represented in tests;
5. benchmark behavior is compared with curated structures and at least one
   independent detector/tool where practical;
6. discrepancies are documented rather than hidden.

## 9. Manual curation

Manual edits must not overwrite automatic evidence.

A manually added interaction should have:

`origin = "manual"`

and preserve who/what added it and, where possible, the geometry at the time of
addition.

## 10. Versioning

Interaction behavior must be reproducible from the rule-set version. Changing a
scientific cutoff requires a rule-set version change and regression tests.
