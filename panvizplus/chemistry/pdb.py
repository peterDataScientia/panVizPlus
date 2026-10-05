"""Conservative PDB normalization for the first panVizPlus vertical slice."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .models import Atom, NormalizedStructure, distance, normalized_bond


_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
    "SE": 1.20,
}


def _guess_element(atom_name: str) -> str:
    stripped = "".join(ch for ch in atom_name.strip() if ch.isalpha()).upper()
    if not stripped:
        return ""
    if len(stripped) >= 2 and stripped[:2] in _COVALENT_RADII:
        return stripped[:2]
    return stripped[0]


def _parse_formal_charge(raw: str) -> int | None:
    value = raw.strip()
    if not value:
        return None
    try:
        if value[-1:] in {"+", "-"}:
            magnitude = int(value[:-1] or "1")
            return magnitude if value[-1] == "+" else -magnitude
        return int(value)
    except ValueError:
        return None


def read_pdb(path: str | Path) -> NormalizedStructure:
    """Read the first PDB model into a normalized structure.

    Blank/A alternate locations are retained; other alternate locations are
    ignored. PDB CONECT records are preserved. Local connectivity inference is
    used only to support geometry and is not bond-order perception.
    """
    path = Path(path)
    atoms: list[Atom] = []
    explicit_bonds: set[tuple[int, int]] = set()
    seen_serials: set[int] = set()
    saw_model = False

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = line[0:6].strip().upper()
        if record == "MODEL":
            if saw_model:
                break
            saw_model = True
            continue
        if record == "ENDMDL" and saw_model:
            break

        if record in {"ATOM", "HETATM"}:
            altloc = line[16:17].strip()
            if altloc not in {"", "A"}:
                continue
            try:
                serial = int(line[6:11])
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip().upper()
                chain_id = line[21:22].strip()
                residue_number = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            if serial in seen_serials:
                continue
            seen_serials.add(serial)
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            element = element or _guess_element(atom_name)
            charge = _parse_formal_charge(line[78:80] if len(line) >= 80 else "")
            atoms.append(
                Atom(
                    atom_id=serial,
                    name=atom_name,
                    element=element,
                    residue_name=residue_name,
                    residue_number=residue_number,
                    chain_id=chain_id,
                    x=x,
                    y=y,
                    z=z,
                    record_type=record,
                    altloc=altloc,
                    formal_charge=charge,
                )
            )
        elif record == "CONECT":
            try:
                source = int(line[6:11])
            except ValueError:
                continue
            for start in range(11, len(line), 5):
                field = line[start : start + 5].strip()
                if not field:
                    continue
                try:
                    target = int(field)
                except ValueError:
                    continue
                if source != target:
                    explicit_bonds.add(normalized_bond(source, target))

    structure = NormalizedStructure(
        atoms=atoms,
        bonds={b for b in explicit_bonds if b[0] in seen_serials and b[1] in seen_serials},
        source_path=str(path),
    )
    _infer_local_connectivity(structure)
    return structure


def _infer_local_connectivity(structure: NormalizedStructure) -> None:
    atoms = structure.atoms
    by_residue: dict[tuple[str, int, str], list[Atom]] = defaultdict(list)
    for atom in atoms:
        by_residue[(atom.chain_id, atom.residue_number, atom.residue_name)].append(atom)

    def maybe_add(a: Atom, b: Atom) -> None:
        ra = _COVALENT_RADII.get(a.element.upper())
        rb = _COVALENT_RADII.get(b.element.upper())
        if ra is None or rb is None:
            return
        d = distance(a, b)
        if 0.4 < d <= ra + rb + 0.45:
            structure.bonds.add(normalized_bond(a.atom_id, b.atom_id))

    for residue_atoms in by_residue.values():
        for i, atom_a in enumerate(residue_atoms):
            for atom_b in residue_atoms[i + 1 :]:
                maybe_add(atom_a, atom_b)

    carbonyl_c = [a for a in atoms if a.record_type == "ATOM" and a.name.upper() == "C"]
    amide_n = [a for a in atoms if a.record_type == "ATOM" and a.name.upper() == "N"]
    for c_atom in carbonyl_c:
        for n_atom in amide_n:
            if c_atom.chain_id != n_atom.chain_id:
                continue
            if n_atom.residue_number != c_atom.residue_number + 1:
                continue
            if distance(c_atom, n_atom) <= 1.8:
                structure.bonds.add(normalized_bond(c_atom.atom_id, n_atom.atom_id))
