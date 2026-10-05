# panVizPlus Interaction Specification

## Contract

panVizPlus separates chemistry perception, interaction detection, provenance, and rendering. Rendering never decides whether an interaction exists.

## Native engine

The source of truth is `panvizplus/interactions/engine.py` plus the versioned registry `panvizplus/rules/panvizplus_v1.yaml`.

Current families are hydrogen bonds, hydrophobic contacts, salt bridges, π–π interactions, π–cation interactions, halogen bonds, metal coordination, water bridges, and unfavorable van der Waals contacts.

## Provenance

Each `InteractionRecord` stores:

- interaction type;
- ligand and protein sites;
- residue name, number, and chain;
- measured distances/angles;
- criterion thresholds and pass/fail values;
- detector and ruleset version;
- chemistry source and confidence;
- automatic/manual origin.

## Chemistry normalization

PDB coordinates do not guarantee bond order, formal charge, aromaticity, or protonation. Any feature inferred from PDB connectivity must carry an explicit confidence/source label. Exact chemical definitions should supersede coordinate-only inference when authoritative inputs become available.

## Rings

Protein aromatic systems use residue templates. Ligand π systems are currently recognized as planar 5- or 6-member cycles from ligand connectivity and are labeled with inference confidence.

## Versioning

Changing a scientific criterion requires a ruleset version change and regression tests. Research defaults must not be described as exact proprietary-tool defaults unless directly documented and independently verified.
