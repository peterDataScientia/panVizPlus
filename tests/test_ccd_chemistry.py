from math import cos, pi, sin

from panvizplus.chemistry.ccd import parse_ccd_cif
from panvizplus.chemistry.models import Atom, NormalizedStructure
from panvizplus.chemistry.rdkit_layer import build_ligand_chemistry


CCD_BENZENE = """data_BNZ
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
BNZ C1 C 0
BNZ C2 C 0
BNZ C3 C 0
BNZ C4 C 0
BNZ C5 C 0
BNZ C6 C 0
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
BNZ C1 C2 AROM Y
BNZ C2 C3 AROM Y
BNZ C3 C4 AROM Y
BNZ C4 C5 AROM Y
BNZ C5 C6 AROM Y
BNZ C6 C1 AROM Y
#
"""


def atom(i, name, xyz):
    return Atom(i, name, "C", "BNZ", 1, "Z", *xyz, "HETATM")


def test_ccd_parser_reads_atom_names_and_aromatic_bonds():
    template = parse_ccd_cif(CCD_BENZENE, source_url="test")
    assert template is not None
    assert template.comp_id == "BNZ"
    assert len(template.atoms) == 6
    assert len(template.bonds) == 6
    assert all(b.aromatic for b in template.bonds)


def test_ccd_template_drives_rdkit_features(monkeypatch):
    template = parse_ccd_cif(CCD_BENZENE, source_url="test")
    monkeypatch.setattr(
        "panvizplus.chemistry.rdkit_layer.fetch_ccd_template",
        lambda comp_id: template if comp_id == "BNZ" else None,
    )

    atoms = []
    for i in range(6):
        angle = 2 * pi * i / 6
        atoms.append(atom(i + 1, f"C{i+1}", (1.4 * cos(angle), 1.4 * sin(angle), 0.0)))

    structure = NormalizedStructure(
        atoms=atoms,
        bonds={(1,2),(2,3),(3,4),(4,5),(5,6),(1,6)},
    )
    chemistry = build_ligand_chemistry(structure, "BNZ:Z:1", net_charge=0)

    assert chemistry.reconstruction_mode == "wwPDB_CCD"
    assert chemistry.confidence == "high"
    assert chemistry.aromatic_rings
    assert "wwPDB_CCD:BNZ" in chemistry.source
