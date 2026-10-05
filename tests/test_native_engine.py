from panvizplus.chemistry.models import Atom, NormalizedStructure
from panvizplus.interactions.engine import analyze_structure


def atom(i, name, element, res, num, chain, xyz, record="ATOM", charge=None):
    return Atom(i, name, element, res, num, chain, *xyz, record, formal_charge=charge)


def test_native_engine_hydrophobic_contact():
    s = NormalizedStructure(
        atoms=[
            atom(1, "C1", "C", "LIG", 1, "Z", (0.0, 0.0, 0.0), "HETATM"),
            atom(2, "C2", "C", "LIG", 1, "Z", (1.5, 0.0, 0.0), "HETATM"),
            atom(10, "CD1", "C", "LEU", 10, "A", (3.7, 0.0, 0.0)),
        ],
        bonds={(1, 2)},
    )
    records = analyze_structure(s, "LIG:Z:1", ligand_net_charge=0)
    assert any(r.interaction_type == "hydrophobic_contact" for r in records)


def test_native_engine_salt_bridge_with_explicit_charge():
    ligand = [
        atom(1, "N1", "N", "LIG", 1, "Z", (0.0, 0.0, 0.0), "HETATM", 1),
        atom(2, "C1", "C", "LIG", 1, "Z", (1.4, 0.0, 0.0), "HETATM"),
        atom(3, "C2", "C", "LIG", 1, "Z", (-1.4, 0.0, 0.0), "HETATM"),
        atom(4, "C3", "C", "LIG", 1, "Z", (0.0, 1.4, 0.0), "HETATM"),
        atom(5, "C4", "C", "LIG", 1, "Z", (0.0, -1.4, 0.0), "HETATM"),
    ]
    s = NormalizedStructure(
        atoms=ligand + [atom(20, "OD1", "O", "ASP", 20, "A", (3.4, 0.0, 0.0), "ATOM", -1)],
        bonds={(1,2),(1,3),(1,4),(1,5)},
    )
    records = analyze_structure(s, "LIG:Z:1", ligand_net_charge=1)
    salt = [r for r in records if r.interaction_type == "salt_bridge"]
    assert len(salt) == 1
    assert salt[0].measurements["distance"] == 3.4


def test_native_engine_halogen_geometry():
    s = NormalizedStructure(
        atoms=[
            atom(1, "C1", "C", "LIG", 1, "Z", (-1.7, 0.0, 0.0), "HETATM"),
            atom(2, "CL1", "CL", "LIG", 1, "Z", (0.0, 0.0, 0.0), "HETATM"),
            atom(3, "OD1", "O", "ASP", 20, "A", (3.0, 0.0, 0.0)),
        ],
        bonds={(1, 2)},
    )
    records = analyze_structure(s, "LIG:Z:1", ligand_net_charge=0)
    xb = [r for r in records if r.interaction_type == "halogen_bond"]
    assert len(xb) == 1
    assert xb[0].measurements["C_X_A_angle"] == 180.0


def test_native_engine_detector_provenance():
    s = NormalizedStructure(
        atoms=[
            atom(1, "C1", "C", "LIG", 1, "Z", (0.0, 0.0, 0.0), "HETATM"),
            atom(2, "C2", "C", "LIG", 1, "Z", (1.5, 0.0, 0.0), "HETATM"),
            atom(10, "CD1", "C", "LEU", 10, "A", (3.6, 0.0, 0.0)),
        ],
        bonds={(1,2)},
    )
    records = analyze_structure(s, "LIG:Z:1", ligand_net_charge=0)
    assert records
    assert all(r.detector == "panVizPlus-native" for r in records)
    assert all(r.ruleset == "panvizplus_v1" for r in records)
