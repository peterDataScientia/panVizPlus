"""RDKit-backed ligand chemistry perception for panVizPlus.

RDKit supplies molecular chemistry when the ligand graph is chemically valid.
panVizPlus does not force bond-order perception on an invalid or incomplete PDB
graph; in that case it degrades safely and asks for authoritative chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os

from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, rdDepictor, rdDetermineBonds
from rdkit.Geometry import Point3D

from .models import Atom, NormalizedStructure, normalized_bond


@dataclass(slots=True)
class LigandChemistry:
    mol: Chem.Mol
    selector: str
    rd_idx_to_atom_id: dict[int, int]
    atom_id_to_rd_idx: dict[int, int]
    donor_atom_ids: set[int] = field(default_factory=set)
    acceptor_atom_ids: set[int] = field(default_factory=set)
    positive_atom_ids: set[int] = field(default_factory=set)
    negative_atom_ids: set[int] = field(default_factory=set)
    hydrophobe_atom_ids: set[int] = field(default_factory=set)
    aromatic_rings: list[tuple[int, ...]] = field(default_factory=list)
    formal_charges: dict[int, int] = field(default_factory=dict)
    bond_orders: dict[tuple[int, int], float] = field(default_factory=dict)
    confidence: str = "high"
    source: str = "rdkit"
    reconstruction_mode: str = "rdkit_bond_orders"
    warnings: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _feature_factory():
    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    return ChemicalFeatures.BuildFeatureFactory(fdef)


def build_ligand_chemistry(
    structure: NormalizedStructure,
    selector: str,
    net_charge: int | None = None,
) -> LigandChemistry:
    atoms = structure.ligand_atoms(selector)
    if not atoms:
        raise ValueError(f"No ligand atoms found for {selector}.")

    atom_ids = {a.atom_id for a in atoms}
    ligand_bonds = {
        normalized_bond(left, right)
        for left, right in structure.bonds
        if left in atom_ids and right in atom_ids
    }
    inferred_charge = sum(a.formal_charge or 0 for a in atoms)
    charge = int(net_charge if net_charge is not None else inferred_charge)

    mol, rd_idx_to_atom_id, atom_id_to_rd_idx = _coordinate_mol(
        atoms, ligand_bonds, preserve_input_charges=True
    )
    warnings: list[str] = []
    confidence = (
        "high"
        if net_charge is not None or any(a.formal_charge is not None for a in atoms)
        else "medium"
    )
    source = "rdkit_BaseFeatures+DetermineBondOrders"
    reconstruction_mode = "rdkit_bond_orders"

    try:
        rdDetermineBonds.DetermineBondOrders(
            mol,
            charge=charge,
            allowChargedFragments=True,
            embedChiral=True,
        )
        chemistry = LigandChemistry(
            mol=mol,
            selector=selector,
            rd_idx_to_atom_id=rd_idx_to_atom_id,
            atom_id_to_rd_idx=atom_id_to_rd_idx,
            confidence=confidence,
            source=source,
            reconstruction_mode=reconstruction_mode,
            warnings=warnings,
        )
        _populate_features(chemistry)
        return chemistry
    except Exception as exc:
        # Heavy-atom PDB files may not contain enough chemistry to recover
        # bond orders reliably. Do not cascade into sanitization/SMARTS errors
        # or invent donor/acceptor/aromatic features from an invalid graph.
        fallback, rd_idx_to_atom_id, atom_id_to_rd_idx = _coordinate_mol(
            atoms, ligand_bonds, preserve_input_charges=True
        )
        fallback.UpdatePropertyCache(strict=False)
        chemistry = LigandChemistry(
            mol=fallback,
            selector=selector,
            rd_idx_to_atom_id=rd_idx_to_atom_id,
            atom_id_to_rd_idx=atom_id_to_rd_idx,
            confidence="low",
            source="pdb_connectivity_only",
            reconstruction_mode="authoritative_chemistry_required",
            warnings=[
                "RDKit could not assign a chemically valid bond-order model from this PDB "
                f"({type(exc).__name__}). Interaction classes that require reliable ligand "
                "donor/acceptor, charge, or aromaticity perception are withheld. "
                "Use the correct ligand net charge and provide CCD/SDF/MOL2 chemistry."
            ],
        )
        _populate_graph_metadata(chemistry)
        return chemistry


def _coordinate_mol(
    atoms: list[Atom],
    bonds: set[tuple[int, int]],
    *,
    preserve_input_charges: bool,
) -> tuple[Chem.Mol, dict[int, int], dict[int, int]]:
    rw = Chem.RWMol()
    rd_idx_to_atom_id: dict[int, int] = {}
    atom_id_to_rd_idx: dict[int, int] = {}

    for atom in atoms:
        rd_atom = Chem.Atom(atom.element.title())
        if preserve_input_charges and atom.formal_charge is not None:
            rd_atom.SetFormalCharge(int(atom.formal_charge))
        idx = rw.AddAtom(rd_atom)
        rd_idx_to_atom_id[idx] = atom.atom_id
        atom_id_to_rd_idx[atom.atom_id] = idx

    for left, right in sorted(bonds):
        a = atom_id_to_rd_idx.get(left)
        b = atom_id_to_rd_idx.get(right)
        if a is None or b is None or a == b:
            continue
        if rw.GetBondBetweenAtoms(a, b) is None:
            rw.AddBond(a, b, Chem.BondType.SINGLE)

    conf = Chem.Conformer(len(atoms))
    atom_lookup = {a.atom_id: a for a in atoms}
    for idx, atom_id in rd_idx_to_atom_id.items():
        atom = atom_lookup[atom_id]
        conf.SetAtomPosition(idx, Point3D(atom.x, atom.y, atom.z))
    rw.AddConformer(conf, assignId=True)
    return rw.GetMol(), rd_idx_to_atom_id, atom_id_to_rd_idx


def _populate_features(chemistry: LigandChemistry) -> None:
    mol = chemistry.mol
    factory = _feature_factory()

    family_to_target = {
        "Donor": chemistry.donor_atom_ids,
        "Acceptor": chemistry.acceptor_atom_ids,
        "PosIonizable": chemistry.positive_atom_ids,
        "NegIonizable": chemistry.negative_atom_ids,
        "Hydrophobe": chemistry.hydrophobe_atom_ids,
        "LumpedHydrophobe": chemistry.hydrophobe_atom_ids,
    }
    for feature in factory.GetFeaturesForMol(mol):
        target = family_to_target.get(feature.GetFamily())
        if target is None:
            continue
        for rd_idx in feature.GetAtomIds():
            atom_id = chemistry.rd_idx_to_atom_id.get(int(rd_idx))
            if atom_id is not None:
                target.add(atom_id)

    _populate_graph_metadata(chemistry)

    for atom in mol.GetAtoms():
        atom_id = chemistry.rd_idx_to_atom_id[atom.GetIdx()]
        charge = int(atom.GetFormalCharge())
        if charge == 0 and atom.GetAtomicNum() == 6:
            heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            if all(n.GetAtomicNum() in {6, 9, 17, 35, 53} for n in heavy_neighbors):
                chemistry.hydrophobe_atom_ids.add(atom_id)
        elif charge == 0 and atom.GetAtomicNum() == 16:
            heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            if heavy_neighbors and all(n.GetAtomicNum() == 6 for n in heavy_neighbors):
                chemistry.hydrophobe_atom_ids.add(atom_id)

    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) >= 5 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            chemistry.aromatic_rings.append(
                tuple(chemistry.rd_idx_to_atom_id[int(i)] for i in ring)
            )


def _populate_graph_metadata(chemistry: LigandChemistry) -> None:
    mol = chemistry.mol
    for atom in mol.GetAtoms():
        atom_id = chemistry.rd_idx_to_atom_id[atom.GetIdx()]
        charge = int(atom.GetFormalCharge())
        chemistry.formal_charges[atom_id] = charge
        if charge > 0:
            chemistry.positive_atom_ids.add(atom_id)
        elif charge < 0:
            chemistry.negative_atom_ids.add(atom_id)

    for bond in mol.GetBonds():
        a_id = chemistry.rd_idx_to_atom_id[bond.GetBeginAtomIdx()]
        b_id = chemistry.rd_idx_to_atom_id[bond.GetEndAtomIdx()]
        chemistry.bond_orders[normalized_bond(a_id, b_id)] = float(
            bond.GetBondTypeAsDouble()
        )


def ligand_2d_coordinates(chemistry: LigandChemistry) -> dict[int, tuple[float, float]]:
    mol = Chem.Mol(chemistry.mol)
    mol.UpdatePropertyCache(strict=False)
    rdDepictor.Compute2DCoords(mol, canonOrient=True)
    conf = mol.GetConformer()
    return {
        chemistry.rd_idx_to_atom_id[idx]: (
            float(conf.GetAtomPosition(idx).x),
            float(conf.GetAtomPosition(idx).y),
        )
        for idx in chemistry.rd_idx_to_atom_id
    }
