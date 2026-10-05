from types import SimpleNamespace

from panvizplus.rendering.native_svg import _residue_positions


def rec(resname, resnum, chain, ligand_site):
    return SimpleNamespace(
        residue_name=resname,
        residue_number=resnum,
        chain_id=chain,
        ligand_site=ligand_site,
    )


def atom(atom_id, name):
    return SimpleNamespace(atom_id=atom_id, name=name)


def test_residue_layout_is_anchor_based_not_circular():
    coords = {
        1: (300.0, 400.0),
        2: (600.0, 400.0),
        3: (450.0, 250.0),
    }
    atom_by_name = {
        "C1": atom(1, "C1"),
        "C2": atom(2, "C2"),
        "N1": atom(3, "N1"),
    }
    records = [
        rec("LEU", 10, "A", "LIG:Z:1:C1"),
        rec("TYR", 20, "A", "LIG:Z:1:C2"),
        rec("SER", 30, "A", "LIG:Z:1:N1"),
    ]
    chemistry = SimpleNamespace(aromatic_rings=[])
    center = (450.0, 400.0)

    pos = _residue_positions(
        records,
        atom_by_name,
        coords,
        center,
        chemistry,
        900,
        700,
    )

    left = pos[("LEU", 10, "A")]
    right = pos[("TYR", 20, "A")]
    top = pos[("SER", 30, "A")]

    # Each label remains on the local side of the ligand interaction anchor.
    assert left[0] < coords[1][0]
    assert right[0] > coords[2][0]
    assert top[1] < coords[3][1]

    # Placement is tied to the local interaction anchor, not to residue order.
    def d(p, q):
        return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5

    assert d(left, coords[1]) < d(left, coords[2])
    assert d(right, coords[2]) < d(right, coords[1])
    assert d(top, coords[3]) < d(top, coords[1])
