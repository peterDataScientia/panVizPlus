"""RDKit-backed ligand chemistry perception for panVizPlus.

RDKit supplies molecular chemistry (bond orders, aromaticity, formal charge,
SMARTS-based pharmacophore features, and 2D depiction). panVizPlus remains
responsible for protein-ligand interaction classification and geometric rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os

from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, rdDepictor, rdDetermineBonds
from rdkit.Geometry import Point3D

from .models import NormalizedStructure, normalized_bond


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
    rw = Chem.RWMol()
    rd_idx_to_atom_id: dict[int, int] = {}
    atom_id_to_rd_idx: dict[int, int] = {}

    for atom in atoms:
        rd_atom = Chem.Atom(atom.element.title())
        if atom.formal_charge is not None:
            rd_atom.SetFormalCharge(int(atom.formal_charge))
        idx = rw.AddAtom(rd_atom)
        rd_idx_to_atom_id[idx] = atom.atom_id
        atom_id_to_rd_idx[atom.atom_id] = idx

    for left, right in sorted(structure.bonds):
        if left in atom_ids and right in atom_ids:
            a = atom_id_to_rd_idx[left]
            b = atom_id_to_rd_idx[right]
            if rw.GetBondBetweenAtoms(a, b) is None:
                rw.AddBond(a, b, Chem.BondType.SINGLE)

    conf = Chem.Conformer(len(atoms))
    for idx, atom_id in rd_idx_to_atom_id.items():
        atom = next(a for a in atoms if a.atom_id == atom_id)
        conf.SetAtomPosition(idx, Point3D(atom.x, atom.y, atom.z))
    rw.AddConformer(conf, assignId=True)

    mol = rw.GetMol()
    warnings: list[str] = []
    inferred_charge = sum(a.formal_charge or 0 for a in atoms)
    charge = int(net_charge if net_charge is not None else inferred_charge)

    try:
        rdDetermineBonds.DetermineBondOrders(
            mol,
            charge=charge,
            allowChargedFragments=True,
            embedChiral=True,
        )
        confidence = "high" if (net_charge is not None or any(a.formal_charge is not None for a in atoms)) else "medium"
        if net_charge is None and not any(a.formal_charge is not None for a in atoms):
            warnings.append(
                "Ligand net charge was not explicitly supplied; RDKit bond-order perception used charge 0."
            )
    except Exception as exc:
        confidence = "low"
        warnings.append(
            f"RDKit bond-order perception failed ({type(exc).__name__}); using connectivity-only chemistry."
        )
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        confidence = "low"
        warnings.append(f"RDKit sanitization warning: {type(exc).__name__}.")

    chemistry = LigandChemistry(
        mol=mol,
        selector=selector,
        rd_idx_to_atom_id=rd_idx_to_atom_id,
        atom_id_to_rd_idx=atom_id_to_rd_idx,
        confidence=confidence,
        source="rdkit_BaseFeatures+DetermineBondOrders",
        warnings=warnings,
    )
    _populate_features(chemistry)
    return chemistry


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
    try:
        for feature in factory.GetFeaturesForMol(mol):
            target = family_to_target.get(feature.GetFamily())
            if target is None:
                continue
            for rd_idx in feature.GetAtomIds():
                atom_id = chemistry.rd_idx_to_atom_id.get(int(rd_idx))
                if atom_id is not None:
                    target.add(atom_id)
    except Exception as exc:
        chemistry.confidence = "low"
        chemistry.warnings.append(f"RDKit feature perception warning: {type(exc).__name__}.")

    for atom in mol.GetAtoms():
        atom_id = chemistry.rd_idx_to_atom_id[atom.GetIdx()]
        charge = int(atom.GetFormalCharge())
        chemistry.formal_charges[atom_id] = charge
        if charge > 0:
            chemistry.positive_atom_ids.add(atom_id)
        elif charge < 0:
            chemistry.negative_atom_ids.add(atom_id)

    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) < 5:
            continue
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            chemistry.aromatic_rings.append(
                tuple(chemistry.rd_idx_to_atom_id[int(i)] for i in ring)
            )

    for bond in mol.GetBonds():
        a_id = chemistry.rd_idx_to_atom_id[bond.GetBeginAtomIdx()]
        b_id = chemistry.rd_idx_to_atom_id[bond.GetEndAtomIdx()]
        chemistry.bond_orders[normalized_bond(a_id, b_id)] = float(bond.GetBondTypeAsDouble())


def ligand_2d_coordinates(chemistry: LigandChemistry) -> dict[int, tuple[float, float]]:
    mol = Chem.Mol(chemistry.mol)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    return {
        chemistry.rd_idx_to_atom_id[idx]: (
            float(conf.GetAtomPosition(idx).x),
            float(conf.GetAtomPosition(idx).y),
        )
        for idx in chemistry.rd_idx_to_atom_id
    }
