"""Normalized molecular structure models used by panVizPlus."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt


COMMON_ION_ELEMENTS = {
    "NA", "K", "LI", "MG", "CA", "ZN", "MN", "FE", "CU", "CO", "NI", "CD",
    "CL", "BR", "I", "CS", "SR", "BA", "HG", "PB",
}


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
        return [a for a in self.atoms if a.residue_name.upper() in {"HOH", "WAT", "H2O"}]

    def ligand_atoms(self, residue_name: str | None = None) -> list[Atom]:
        atoms = [
            a
            for a in self.atoms
            if a.record_type == "HETATM"
            and a.residue_name.upper() not in {"HOH", "WAT", "H2O"}
            and a.element.upper() not in COMMON_ION_ELEMENTS
        ]
        if residue_name:
            atoms = [a for a in atoms if a.residue_name.upper() == residue_name.upper()]
        return atoms


def distance(a: Atom, b: Atom) -> float:
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def normalized_bond(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)
