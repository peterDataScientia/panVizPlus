"""Normalized molecular structure models used by panVizPlus."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import sqrt


COMMON_ION_ELEMENTS = {
    "NA", "K", "LI", "MG", "CA", "ZN", "MN", "FE", "CU", "CO", "NI", "CD",
    "CL", "BR", "I", "CS", "SR", "BA", "HG", "PB",
}
WATER_NAMES = {"HOH", "WAT", "H2O"}


@dataclass(slots=True)
class Atom:
    atom_id: int
    name: str
    element: str
    residue_name: str
    residue_number: int
    chain_id: str
    x: float
    y: float
    z: float
    record_type: str
    altloc: str = ""
    formal_charge: int | None = None

    @property
    def coord(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() in {"H", "D"}

    @property
    def residue_key(self) -> tuple[str, str, int]:
        return (self.residue_name, self.chain_id, self.residue_number)


@dataclass(slots=True)
class NormalizedStructure:
    atoms: list[Atom]
    bonds: set[tuple[int, int]] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    source_path: str | None = None

    def atom_map(self) -> dict[int, Atom]:
        return {atom.atom_id: atom for atom in self.atoms}

    def neighbors(self, atom_id: int) -> list[Atom]:
        amap = self.atom_map()
        out: list[Atom] = []
        for left, right in self.bonds:
            if left == atom_id and right in amap:
                out.append(amap[right])
            elif right == atom_id and left in amap:
                out.append(amap[left])
        return out

    def protein_atoms(self) -> list[Atom]:
        return [a for a in self.atoms if a.record_type == "ATOM"]

    def water_atoms(self) -> list[Atom]:
        return [a for a in self.atoms if a.residue_name.upper() in WATER_NAMES]

    def ligand_atoms(self, selector: str | None = None) -> list[Atom]:
        het = [
            a for a in self.atoms
            if a.record_type == "HETATM" and a.residue_name.upper() not in WATER_NAMES
        ]
        counts = Counter(a.residue_key for a in het)
        atoms = [
            a for a in het
            if not (
                counts[a.residue_key] == 1
                and a.element.upper() in COMMON_ION_ELEMENTS
            )
        ]
        if selector:
            parts = str(selector).split(":")
            residue_name = parts[0].upper()
            atoms = [a for a in atoms if a.residue_name.upper() == residue_name]
            if len(parts) >= 2 and parts[1] not in {"", "-"}:
                atoms = [a for a in atoms if a.chain_id == parts[1]]
            if len(parts) >= 3 and parts[2]:
                try:
                    residue_number = int(parts[2])
                    atoms = [a for a in atoms if a.residue_number == residue_number]
                except ValueError:
                    pass
        return atoms

    def ligand_residues(self) -> list[tuple[str, str, int]]:
        seen = {a.residue_key for a in self.ligand_atoms()}
        return sorted(seen, key=lambda x: (x[0], x[1], x[2]))

    def ion_atoms(self) -> list[Atom]:
        het = [
            a for a in self.atoms
            if a.record_type == "HETATM" and a.residue_name.upper() not in WATER_NAMES
        ]
        counts = Counter(a.residue_key for a in het)
        return [
            a for a in het
            if counts[a.residue_key] == 1 and a.element.upper() in COMMON_ION_ELEMENTS
        ]


def distance(a: Atom, b: Atom) -> float:
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def normalized_bond(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)
