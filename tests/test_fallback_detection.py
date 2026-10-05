from math import cos, pi, sin

from rdkit import Chem

from panvizplus.chemistry.models import Atom, NormalizedStructure
from panvizplus.chemistry.rdkit_layer import LigandChemistry
from panvizplus.interactions import engine


def atom(i, name, element, res, num, chain, xyz, record="HETATM"):
    return Atom(i, name, element, res, num, chain, *xyz, record)


def unresolved_chemistry(selector: str) -> LigandChemistry:
    return LigandChemistry(
        mol=Chem.Mol(),
        selector=selector,
        rd_idx_to_atom_id={},
        atom_id_to_rd_idx={},
        confidence="low",
        source="pdb_connectivity_only",
        reconstruction_mode="authoritative_chemistry_required",
        warnings=["test unresolved chemistry"],
    )


def test_unresolved_chemistry_does_not_disable_hydrophobic_detector(monkeypatch):
    s = NormalizedStructure(
        atoms=[
            atom(1, "C1", "C", "LIG", 1, "Z", (0.0, 0.0, 0.0)),
            atom(2, "C2", "C", "LIG", 1, "Z", (1.5, 0.0, 0.0)),
            atom(10, "CD1", "C", "LEU", 10, "A", (3.7, 0.0, 0.0), "ATOM"),
        ],
        bonds={(1, 2)},
    )
    monkeypatch.setattr(
        engine,
        "build_ligand_chemistry",
        lambda structure, selector, net_charge=None: unresolved_chemistry(selector),
    )

    records = engine.analyze_structure(s, "LIG:Z:1", ligand_net_charge=0)
    hyd = [r for r in records if r.interaction_type == "hydrophobic_contact"]

    assert hyd
    assert hyd[0].metadata["chemistry_source"] == "pdb_connectivity_hydrophobe_fallback"


def test_unresolved_chemistry_does_not_disable_hbond_detector(monkeypatch):
    s = NormalizedStructure(
        atoms=[
            atom(1, "N1", "N", "LIG", 1, "Z", (0.0, 0.0, 0.0)),
            atom(2, "C1", "C", "LIG", 1, "Z", (-1.3, 0.0, 0.0)),
            atom(3, "C2", "C", "LIG", 1, "Z", (0.0, -1.3, 0.0)),
            atom(10, "OG", "O", "SER", 205, "A", (2.9, 0.0, 0.0), "ATOM"),
            atom(11, "CB", "C", "SER", 205, "A", (4.2, 0.0, 0.0), "ATOM"),
        ],
        bonds={(1,2),(1,3),(10,11)},
    )
    monkeypatch.setattr(
        engine,
        "build_ligand_chemistry",
        lambda structure, selector, net_charge=None: unresolved_chemistry(selector),
    )

    records = engine.analyze_structure(s, "LIG:Z:1", ligand_net_charge=0)
    hb = [r for r in records if r.interaction_type == "conventional_hbond"]

    assert hb
    assert hb[0].ligand_site.endswith(":N1")
    assert "pdb_" in str(
        hb[0].metadata.get("ligand_feature_source")
        or hb[0].metadata.get("chemistry_source")
    )


def test_unresolved_chemistry_does_not_disable_pi_detector(monkeypatch):
    atoms = []
    ligand_bonds = set()
    radius = 1.40

    for i in range(6):
        angle = 2 * pi * i / 6
        atoms.append(
            atom(
                i + 1,
                f"C{i+1}",
                "C",
                "LIG",
                1,
                "Z",
                (radius * cos(angle), radius * sin(angle), 0.0),
            )
        )
        ligand_bonds.add((i + 1, ((i + 1) % 6) + 1))

    phe_names = ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]
    for i, name in enumerate(phe_names):
        angle = 2 * pi * i / 6
        atoms.append(
            atom(
                20 + i,
                name,
                "C",
                "PHE",
                50,
                "A",
                (radius * cos(angle), radius * sin(angle), 3.6),
                "ATOM",
            )
        )

    s = NormalizedStructure(atoms=atoms, bonds=ligand_bonds)
    monkeypatch.setattr(
        engine,
        "build_ligand_chemistry",
        lambda structure, selector, net_charge=None: unresolved_chemistry(selector),
    )

    records = engine.analyze_structure(s, "LIG:Z:1", ligand_net_charge=0)
    pipi = [r for r in records if r.interaction_type == "pi_pi_stacked"]

    assert pipi
    assert pipi[0].metadata["chemistry_source"] == "planar_5_6_member_cycle_from_connectivity"
