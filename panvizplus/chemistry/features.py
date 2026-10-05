"""Chemical feature perception for panVizPlus native interaction analysis."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Atom, NormalizedStructure, distance


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
    ligand_residue_name: str | None = None,
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

    for atom in structure.ligand_atoms(ligand_residue_name):
        if atom.is_hydrogen:
            continue
        element = atom.element.upper()
        heavy_neighbors = [n for n in structure.neighbors(atom.atom_id) if not n.is_hydrogen]
        hydrogen_ids = tuple(h.atom_id for h in _attached_hydrogens(structure, atom))
        neighbor_ids = tuple(n.atom_id for n in heavy_neighbors)
        charge = atom.formal_charge or 0

        # PDB connectivity alone cannot reliably distinguish, for example,
        # hydroxyl O from carbonyl O. Therefore a ligand donor is confirmed only
        # when an attached hydrogen is explicitly represented. Missing-H donor
        # perception will be upgraded later from authoritative bond-order/
        # protonation data rather than guessed from coordination number.
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
                "pdb_element_connectivity_inference", hydrogen_ids, neighbor_ids,
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
