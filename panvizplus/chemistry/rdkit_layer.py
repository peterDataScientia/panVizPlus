"""RDKit-backed ligand chemistry perception for panVizPlus.

RDKit supplies molecular chemistry (connectivity/bond-order recovery, aromaticity,
formal charge, pharmacophore features, and 2D depiction). panVizPlus remains
responsible for protein-ligand interaction classification and geometric rules.
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
    reconstruction_mode: str = "pdb_connectivity"
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
    confidence = (
        "high"
        if net_charge is not None or any(a.formal_charge is not None for a in atoms)
        else "medium"
    )
    warnings: list[str] = []

    mol, rd_idx_to_atom_id, atom_id_to_rd_idx = _coordinate_mol(atoms, ligand_bonds)
    reconstruction_mode = "pdb_connectivity"
    source = "rdkit_BaseFeatures+DetermineBondOrders"

    try:
        rdDetermineBonds.DetermineBondOrders(
            mol,
            charge=charge,
            allowChargedFragments=True,
            embedChiral=True,
        )
        Chem.SanitizeMol(mol)
    except Exception as first_exc:
        # PDB CONECT records and distance-derived local bonds can occasionally
        # over-connect a ligand. Rebuild the graph from 3D coordinates instead
        # of asking bond-order perception to rescue an impossible graph.
        rebuilt, rd_idx_to_atom_id, atom_id_to_rd_idx = _coordinate_mol(atoms, set())
        try:
            rdDetermineBonds.DetermineBonds(
                rebuilt,
                charge=charge,
                covFactor=1.3,
                allowChargedFragments=True,
                embedChiral=True,
                useVdw=True,
            )
            Chem.SanitizeMol(rebuilt)
            mol = rebuilt
            reconstruction_mode = "rdkit_3d_connectivity_rebuild"
            source = "rdkit_BaseFeatures+DetermineBonds_from_3D"
        except Exception as second_exc:
            # Keep the app operational, but do not run SMARTS chemistry on an
            # invalid graph. Downstream feature sets remain intentionally sparse.
            mol, rd_idx_to_atom_id, atom_id_to_rd_idx = _coordinate_mol(
                atoms, ligand_bonds
            )
            mol.UpdatePropertyCache(strict=False)
            try:
                Chem.GetSymmSSSR(mol)
            except Exception:
                pass
            confidence = "low"
            reconstruction_mode = "connectivity_only_fallback"
            source = "pdb_connectivity_only"
            warnings.append(
                "Ligand chemistry could not be reconstructed reliably from this PDB "
                f"({type(first_exc).__name__}; 3D rebuild: {type(second_exc).__name__}). "
                "Verify the ligand net charge or provide SDF/MOL2/CCD chemistry."
            )

    if net_charge is None and not any(a.formal_charge is not None for a in atoms):
        warnings.append(
            "Ligand net charge was not explicitly supplied; chemistry perception used charge 0."
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

    if reconstruction_mode != "connectivity_only_fallback":
        _populate_features(chemistry)
    else:
        _populate_graph_metadata(chemistry)
    return chemistry


def _coordinate_mol(
    atoms: list[Atom],
    bonds: set[tuple[int, int]],
) -> tuple[Chem.Mol, dict[int, int], dict[int, int]]:
    rw = Chem.RWMol()
    rd_idx_to_atom_id: dict[int, int] = {}
    atom_id_to_rd_idx: dict[int, int] = {}

    for atom in atoms:
        rd_atom = Chem.Atom(atom.element.title())
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

        # BaseFeatures intentionally does not label every aliphatic carbon as a
        # hydrophobe. Augment from the sanitized RDKit molecular graph.
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
        if len(ring) < 5:
            continue
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
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
    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:
        # Even a connectivity-only fallback should remain drawable.
        mol.UpdatePropertyCache(strict=False)
        rdDepictor.Compute2DCoords(mol, canonOrient=False)
    conf = mol.GetConformer()
    return {
        chemistry.rd_idx_to_atom_id[idx]: (
            float(conf.GetAtomPosition(idx).x),
            float(conf.GetAtomPosition(idx).y),
        )
        for idx in chemistry.rd_idx_to_atom_id
    }
