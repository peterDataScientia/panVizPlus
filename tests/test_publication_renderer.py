from math import cos, pi, sin

from panvizplus.chemistry.models import Atom, NormalizedStructure
from panvizplus.rendering.native_svg import render_interaction_svg


def atom(i, name, element, xyz):
    return Atom(i, name, element, "LIG", 1, "Z", *xyz, "HETATM")


def test_publication_renderer_uses_skeletal_rdkit_svg_not_atom_nodes():
    atoms = []
    for i in range(6):
        angle = 2 * pi * i / 6
        atoms.append(
            atom(
                i + 1,
                f"C{i+1}",
                "C",
                (1.40 * cos(angle), 1.40 * sin(angle), 0.0),
            )
        )
    structure = NormalizedStructure(
        atoms=atoms,
        bonds={(1,2),(2,3),(3,4),(4,5),(5,6),(1,6)},
    )
    svg = render_interaction_svg(
        structure,
        "LIG:Z:1",
        [],
        width=900,
        height=600,
        ligand_net_charge=0,
    )

    assert "bond-0" in svg
    assert "atom-0" in svg
    assert "<circle" not in svg
    assert 'fill="white"' in svg
