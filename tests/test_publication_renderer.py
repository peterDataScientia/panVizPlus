from math import cos, pi, sin

from panvizplus.chemistry.models import Atom, NormalizedStructure
from panvizplus.interactions.models import CriterionResult, InteractionRecord
from panvizplus.rendering.native_svg import render_interaction_svg


def atom(i, name, element, xyz):
    return Atom(i, name, element, "LIG", 1, "Z", *xyz, "HETATM")


def _benzene_structure():
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
    return NormalizedStructure(
        atoms=atoms,
        bonds={(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)},
    )


def test_publication_renderer_uses_skeletal_ligand_without_carbon_nodes():
    svg = render_interaction_svg(
        _benzene_structure(),
        "LIG:Z:1",
        [],
        width=900,
        height=600,
        ligand_net_charge=0,
    )

    assert 'data-renderer="panviz-publication"' in svg
    assert "bond-0" in svg
    assert "atom-0" in svg
    assert "pv-residue-bubble" not in svg
    assert "pv-carbon-node" not in svg
    assert 'fill="#FFFFFF"' in svg


def test_publication_renderer_adapts_established_panviz_visual_grammar():
    record = InteractionRecord(
        interaction_id="HYD-0001",
        interaction_type="hydrophobic_contact",
        ligand_site="LIG:Z:1:C1",
        protein_site="LEU:A:210:CD1",
        residue_name="LEU",
        residue_number=210,
        chain_id="A",
        criteria=[
            CriterionResult("atom_distance", 3.8, "<=", 4.5, "angstrom", True)
        ],
        measurements={"distance": 3.8},
    )

    svg = render_interaction_svg(
        _benzene_structure(),
        "LIG:Z:1",
        [record],
        width=1000,
        height=700,
        ligand_net_charge=0,
    )

    assert 'class="pv-residue-bubble"' in svg
    assert "#0AFFEF" in svg
    assert "#595959" in svg
    assert "#D100A0" in svg
    assert 'stroke-width="3.0"' in svg
    assert 'stroke-dasharray="9 5"' in svg
    assert 'class="pv-legend-box"' in svg
    assert "Hydrophobic contact" in svg
    assert ">LEU<" in svg
    assert ">210:A<" in svg
