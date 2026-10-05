from pathlib import Path

from panvizplus.analysis import analyze_pdb
from panvizplus.chemistry.pdb import read_pdb
from panvizplus.interactions.hydrogen_bond import detect_conventional_hbonds


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "complex.pdb"
    path.write_text(text)
    return path


def test_explicit_hydrogen_hbond(tmp_path):
    pdb = """ATOM      1  CA  SER A  10      -1.400   0.000   0.000  1.00 20.00           C
ATOM      2  OG  SER A  10       0.000   0.000   0.000  1.00 20.00           O
ATOM      3  HG  SER A  10       0.960   0.000   0.000  1.00 20.00           H
HETATM    4  O1  LIG B   1       2.800   0.000   0.000  1.00 20.00           O
HETATM    5  C1  LIG B   1       3.900   0.000   0.000  1.00 20.00           C
CONECT    2    3
CONECT    4    5
END
"""
    hbonds = detect_conventional_hbonds(read_pdb(_write(tmp_path, pdb)), "LIG")
    assert len(hbonds) == 1
    hit = hbonds[0]
    assert hit.metadata["geometry_mode"] == "explicit_hydrogen"
    assert hit.measurements["donor_acceptor_distance"] == 2.8
    assert hit.measurements["DHA_angle"] == 180.0
    assert hit.passes_all_criteria


def test_missing_hydrogen_surrogate(tmp_path):
    pdb = """ATOM      1  CE  LYS A  20      -1.400   0.000   0.000  1.00 20.00           C
ATOM      2  NZ  LYS A  20       0.000   0.000   0.000  1.00 20.00           N
HETATM    3  O1  LIG B   1       2.900   0.000   0.000  1.00 20.00           O
HETATM    4  C1  LIG B   1       4.000   0.000   0.000  1.00 20.00           C
CONECT    1    2
CONECT    3    4
END
"""
    hbonds = detect_conventional_hbonds(read_pdb(_write(tmp_path, pdb)), "LIG")
    assert len(hbonds) == 1
    assert hbonds[0].measurements["XDA_angle"] == 180.0
    assert hbonds[0].measurements["DAY_angle"] == 180.0


def test_bad_angle_is_rejected(tmp_path):
    pdb = """ATOM      1  CE  LYS A  20       1.400   0.000   0.000  1.00 20.00           C
ATOM      2  NZ  LYS A  20       0.000   0.000   0.000  1.00 20.00           N
HETATM    3  O1  LIG B   1       2.900   0.000   0.000  1.00 20.00           O
HETATM    4  C1  LIG B   1       4.000   0.000   0.000  1.00 20.00           C
CONECT    1    2
CONECT    3    4
END
"""
    assert detect_conventional_hbonds(read_pdb(_write(tmp_path, pdb)), "LIG") == []


def test_analysis_surfaces_pdb_chemistry_warning(tmp_path):
    pdb = """ATOM      1  CA  SER A  10      -1.400   0.000   0.000  1.00 20.00           C
HETATM    2  O1  LIG B   1       2.800   0.000   0.000  1.00 20.00           O
END
"""
    _, warnings = analyze_pdb(_write(tmp_path, pdb), "LIG")
    assert any("bond orders" in warning for warning in warnings)
