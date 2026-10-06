from math import cos, pi

from panvizplus.chemistry.models import Atom, NormalizedStructure
from panvizplus.interactions.models import CriterionResult, InteractionRecord
from panvizplus.rendering.scene import build_editor_scene, render_editor_html


def ligand_atom(i, name, xyz):
    return Atom(i, name, "C", "LIG", 1, "Z", *xyz, "HETATM")


def benzene():
    atoms = []
    for i in range(6):
        angle = 2 * pi * i / 6
        atoms.append(
            ligand_atom(
                i + 1,
                f"C{i+1}",
                (1.40 * cos(angle), 1.40 * __import__("math").sin(angle), 0.0),
            )
        )
    return NormalizedStructure(
        atoms=atoms,
        bonds={(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)},
    )


def record():
    return InteractionRecord(
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


def test_editor_scene_keeps_panviz_presentation_defaults():
    scene = build_editor_scene(
        benzene(),
        "LIG:Z:1",
        [record()],
        width=1200,
        height=850,
        ligand_net_charge=0,
    )

    assert scene["style"]["bondWidth"] == 2.0
    assert scene["style"]["interactionWidth"] == 3.0
    assert scene["style"]["residue"]["shape"] == "bubble"
    assert scene["style"]["residue"]["nodeSize"] == 60
    assert scene["style"]["residue"]["bubbleColor"] == "#0AFFEF"
    assert scene["style"]["distance"]["textColor"] == "#D100A0"

    interaction = scene["interactions"][0]
    assert interaction["type"] == "HPI"
    assert interaction["color"] == "#595959"
    assert interaction["dash"] == "9 5"
    assert interaction["lineWidth"] == 3.0

    assert scene["labels"][0]["nodeShape"] == "bubble"
    assert scene["legend"]["items"][0]["label"] == "Hydrophobic contact"
    assert scene["scientificData"]["detector"] == "panVizPlus-native"
    assert scene["metadata"]["moleculeLocked"] is True
    assert scene["metadata"]["annotationOnly"] is True


def test_editor_html_preserves_publication_editor_controls():
    scene = build_editor_scene(
        benzene(),
        "LIG:Z:1",
        [record()],
        width=1000,
        height=700,
        ligand_net_charge=0,
    )
    html = render_editor_html(scene)

    for label in (
        "Move",
        "Select",
        "Undo",
        "Redo",
        "Reset",
        "Add interaction",
        "Add distance",
        "Add text",
        "Add arrow",
        "Load layout",
        "Save layout",
        "Export SVG",
        "Export PNG",
        "Export PDF",
        "Show interaction records",
    ):
        assert label in html

    assert "__PANVIZ_SCENE__" not in html
    assert "panVizPlus-native" in html
    assert "3D glass bubble" in html
