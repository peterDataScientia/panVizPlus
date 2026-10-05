"""Chemical feature perception for panVizPlus native interaction analysis."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Atom, NormalizedStructure, distance
from .rdkit_layer import LigandChemistry


@dataclass(frozen=True, slots=True)
class ChemicalFeature:
    atom_id: int
    kind: str
    confidence: str
    source: str
    hydrogen_ids: tuple[int, ...] = ()
    neighbor_ids: tuple[int, ...] = ()


_PROTEIN_DONORS = {
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("ASN", "ND2"), ("GLN", "NE2"),
    ("HIS", "ND1"), ("HIS", "NE2"),
    ("LYS", "NZ"), ("SER", "OG"), ("THR", "OG1"),
    ("TRP", "NE1"), ("TYR", "OH"), ("CYS", "SG"),
}
_PROTEIN_ACCEPTORS = {
    ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
    ("ASN", "OD1"), ("GLN", "OE1"), ("HIS", "ND1"), ("HIS", "NE2"),
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("CYS", "SG"),
}


def perceive_hbond_features(
    structure: NormalizedStructure,
    ligand_selector: str | None = None,
    ligand_chemistry: LigandChemistry | None = None,
) -> tuple[list[ChemicalFeature], list[ChemicalFeature]]:
    """Return protein and ligand donor/acceptor features with provenance."""
    protein_features: list[ChemicalFeature] = []
    ligand_features: list[ChemicalFeature] = []

    for atom in structure.protein_atoms():
        if atom.is_hydrogen:
            continue
        res = atom.residue_name.upper()
        name = atom.name.upper()
        hydrogen_ids = tuple(h.atom_id for h in _attached_hydrogens(structure, atom))
        neighbor_ids = tuple(n.atom_id for n in structure.neighbors(atom.atom_id) if not n.is_hydrogen)

        is_donor = (name == "N" and res != "PRO") or (res, name) in _PROTEIN_DONORS
        is_acceptor = name in {"O", "OXT"} or (res, name) in _PROTEIN_ACCEPTORS

        if is_donor:
            confidence = "conditional" if res == "HIS" and not hydrogen_ids else "high"
            protein_features.append(ChemicalFeature(
                atom.atom_id, "hydrogen_donor", confidence, "protein_residue_template",
                hydrogen_ids, neighbor_ids,
            ))
        if is_acceptor:
            confidence = "conditional" if res == "HIS" else "high"
            protein_features.append(ChemicalFeature(
                atom.atom_id, "hydrogen_acceptor", confidence, "protein_residue_template",
                hydrogen_ids, neighbor_ids,
            ))

    if ligand_chemistry is not None:
        amap = structure.atom_map()
        for atom_id in sorted(ligand_chemistry.donor_atom_ids):
            atom = amap.get(atom_id)
            if atom is None:
                continue
            ligand_features.append(ChemicalFeature(
                atom_id,
                "hydrogen_donor",
                ligand_chemistry.confidence,
                ligand_chemistry.source,
                tuple(h.atom_id for h in _attached_hydrogens(structure, atom)),
                tuple(n.atom_id for n in structure.neighbors(atom_id) if not n.is_hydrogen),
            ))
        for atom_id in sorted(ligand_chemistry.acceptor_atom_ids):
            atom = amap.get(atom_id)
            if atom is None:
                continue
            ligand_features.append(ChemicalFeature(
                atom_id,
                "hydrogen_acceptor",
                ligand_chemistry.confidence,
                ligand_chemistry.source,
                tuple(h.atom_id for h in _attached_hydrogens(structure, atom)),
                tuple(n.atom_id for n in structure.neighbors(atom_id) if not n.is_hydrogen),
            ))

        # A resolved RDKit/CCD graph is authoritative. An unresolved heavy-atom
        # PDB graph, however, must not disable the detector: continue below to
        # the conservative PDB feature fallback and mark its provenance.
        if (
            ligand_features
            or ligand_chemistry.reconstruction_mode != "authoritative_chemistry_required"
        ):
            return protein_features, ligand_features

    # Conservative fallback for unresolved PDB chemistry.
    for atom in structure.ligand_atoms(ligand_selector):
        if atom.is_hydrogen:
            continue
        element = atom.element.upper()
        heavy_neighbors = [n for n in structure.neighbors(atom.atom_id) if not n.is_hydrogen]
        hydrogen_ids = tuple(h.atom_id for h in _attached_hydrogens(structure, atom))
        neighbor_ids = tuple(n.atom_id for n in heavy_neighbors)
        charge = atom.formal_charge or 0

        if element in {"N", "O", "S"} and hydrogen_ids and charge >= 0:
            ligand_features.append(ChemicalFeature(
                atom.atom_id, "hydrogen_donor", "medium",
                "pdb_explicit_hydrogen", hydrogen_ids, neighbor_ids,
            ))

        acceptor = False
        if element == "O" and charge <= 0:
            acceptor = len(heavy_neighbors) <= 2
        elif element == "N" and charge <= 0:
            acceptor = len(heavy_neighbors) <= 2 and not hydrogen_ids
        elif element == "S" and charge <= 0:
            acceptor = len(heavy_neighbors) <= 2

        if acceptor:
            ligand_features.append(ChemicalFeature(
                atom.atom_id, "hydrogen_acceptor", "low",
                "pdb_element_connectivity_fallback", hydrogen_ids, neighbor_ids,
            ))

    return protein_features, ligand_features


def _attached_hydrogens(structure: NormalizedStructure, atom: Atom) -> list[Atom]:
    bonded = [n for n in structure.neighbors(atom.atom_id) if n.is_hydrogen]
    if bonded:
        return bonded
    return [
        h for h in structure.atoms
        if h.is_hydrogen
        and h.chain_id == atom.chain_id
        and h.residue_number == atom.residue_number
        and distance(atom, h) <= 1.25
    ]
