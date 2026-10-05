from math import cos, pi, sin

from panvizplus.chemistry.models import Atom, NormalizedStructure
from panvizplus.chemistry.rdkit_layer import build_ligand_chemistry


def atom(i, name, element, xyz, charge=None):
    return Atom(i, name, element, "LIG", 1, "Z", *xyz, "HETATM", formal_charge=charge)


def test_rdkit_recognizes_aromatic_ring_from_connectivity():
    atoms = []
    for i in range(6):
        angle = 2 * pi * i / 6
        atoms.append(atom(i + 1, f"C{i+1}", "C", (1.4 * cos(angle), 1.4 * sin(angle), 0.0)))
    bonds = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)}
    chemistry = build_ligand_chemistry(
        NormalizedStructure(atoms=atoms, bonds=bonds),
        "LIG:Z:1",
        net_charge=0,
    )
    assert chemistry.aromatic_rings
    assert set(chemistry.aromatic_rings[0]) == {1, 2, 3, 4, 5, 6}
    assert any(abs(order - 1.5) < 1e-6 for order in chemistry.bond_orders.values())


def test_rdkit_amide_feature_perception():
    s = NormalizedStructure(
        atoms=[
            atom(1, "C1", "C", (-1.5, 0.0, 0.0)),
            atom(2, "C2", "C", (0.0, 0.0, 0.0)),
            atom(3, "O1", "O", (1.2, 0.7, 0.0)),
            atom(4, "N1", "N", (1.2, -0.7, 0.0)),
        ],
        bonds={(1, 2), (2, 3), (2, 4)},
    )
    chemistry = build_ligand_chemistry(s, "LIG:Z:1", net_charge=0)
    assert 3 in chemistry.acceptor_atom_ids
    assert 4 in chemistry.donor_atom_ids
    assert 4 not in chemistry.acceptor_atom_ids


def test_rdkit_preserves_quaternary_ammonium_charge():
    atoms = [
        atom(1, "N1", "N", (0.0, 0.0, 0.0), charge=1),
        atom(2, "C1", "C", (1.5, 0.0, 0.0)),
        atom(3, "C2", "C", (-1.5, 0.0, 0.0)),
        atom(4, "C3", "C", (0.0, 1.5, 0.0)),
        atom(5, "C4", "C", (0.0, -1.5, 0.0)),
    ]
    chemistry = build_ligand_chemistry(
        NormalizedStructure(atoms=atoms, bonds={(1,2),(1,3),(1,4),(1,5)}),
        "LIG:Z:1",
        net_charge=1,
    )
    assert chemistry.formal_charges[1] == 1
    assert 1 in chemistry.positive_atom_ids
