"""wwPDB Chemical Component Dictionary support for panVizPlus.

The CCD provides authoritative atom names, formal charges and bond orders for
standard PDB chemical components. This module intentionally parses only the
chem_comp_atom and chem_comp_bond loops needed by the ligand chemistry layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import shlex
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class CCDAtom:
    atom_id: str
    element: str
    charge: int


@dataclass(frozen=True, slots=True)
class CCDBond:
    atom_id_1: str
    atom_id_2: str
    order: str
    aromatic: bool


@dataclass(frozen=True, slots=True)
class CCDTemplate:
    comp_id: str
    atoms: dict[str, CCDAtom]
    bonds: tuple[CCDBond, ...]
    source_url: str


_GENERIC_COMPONENT_IDS = {"LIG", "UNL", "UNK", "DRG", "MOL"}


def is_ccd_candidate(comp_id: str) -> bool:
    cid = comp_id.strip().upper()
    return bool(cid) and cid not in _GENERIC_COMPONENT_IDS and len(cid) <= 5


@lru_cache(maxsize=128)
def fetch_ccd_template(comp_id: str, timeout: float = 5.0) -> CCDTemplate | None:
    cid = comp_id.strip().upper()
    if not is_ccd_candidate(cid):
        return None

    first = cid[0]
    urls = (
        f"https://files.rcsb.org/ligands/download/{cid}.cif",
        f"https://files.rcsb.org/pub/pdb/refdata/chem_comp/{first}/{cid}/{cid}.cif",
    )
    for url in urls:
        try:
            request = Request(
                url,
                headers={"User-Agent": "panVizPlus/0.3 (+https://github.com/peterDataScientia/panVizPlus)"},
            )
            with urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
            template = parse_ccd_cif(text, source_url=url)
            if template and template.comp_id == cid:
                return template
        except (HTTPError, URLError, TimeoutError, ValueError):
            continue
    return None


def parse_ccd_cif(text: str, source_url: str = "embedded") -> CCDTemplate | None:
    loops = _parse_loops(text)
    atom_rows = loops.get("_chem_comp_atom", [])
    bond_rows = loops.get("_chem_comp_bond", [])
    if not atom_rows:
        return None

    comp_id = str(atom_rows[0].get("comp_id", "")).upper()
    atoms: dict[str, CCDAtom] = {}
    for row in atom_rows:
        atom_id = _clean(row.get("atom_id"))
        element = _clean(row.get("type_symbol")).upper()
        if not atom_id or not element:
            continue
        charge = _as_int(row.get("charge"), default=0)
        atoms[atom_id] = CCDAtom(atom_id, element, charge)

    bonds: list[CCDBond] = []
    for row in bond_rows:
        a1 = _clean(row.get("atom_id_1"))
        a2 = _clean(row.get("atom_id_2"))
        order = _clean(row.get("value_order")).upper() or "SING"
        aromatic_flag = _clean(row.get("pdbx_aromatic_flag")).upper()
        if a1 and a2:
            bonds.append(
                CCDBond(
                    a1,
                    a2,
                    order,
                    aromatic=(order == "AROM" or aromatic_flag == "Y"),
                )
            )

    if not comp_id or not atoms or not bonds:
        return None
    return CCDTemplate(comp_id, atoms, tuple(bonds), source_url)


def _parse_loops(text: str) -> dict[str, list[dict[str, str]]]:
    lines = text.splitlines()
    loops: dict[str, list[dict[str, str]]] = {}
    i = 0
    while i < len(lines):
        if lines[i].strip().lower() != "loop_":
            i += 1
            continue
        i += 1
        headers: list[str] = []
        while i < len(lines) and lines[i].lstrip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1
        if not headers:
            continue

        category = headers[0].split(".", 1)[0]
        names = [
            h.split(".", 1)[1] if "." in h else h.lstrip("_")
            for h in headers
        ]
        rows: list[dict[str, str]] = []
        tokens: list[str] = []

        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#"):
                i += 1
                if tokens:
                    break
                continue
            if stripped.lower() == "loop_" or stripped.startswith("_") or stripped.lower().startswith("data_"):
                break
            try:
                tokens.extend(shlex.split(stripped, posix=True))
            except ValueError:
                tokens.extend(stripped.split())
            while len(tokens) >= len(names):
                row_tokens = tokens[: len(names)]
                tokens = tokens[len(names) :]
                rows.append(dict(zip(names, row_tokens)))
            i += 1

        if category in {"_chem_comp_atom", "_chem_comp_bond"}:
            loops.setdefault(category, []).extend(rows)
    return loops


def _clean(value) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value in {".", "?"} else value


def _as_int(value, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default
