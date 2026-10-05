from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_METADATA = {"id", "version", "status", "exact_biovia_reproduction"}

_RULESET_FILES = {
    "panvizplus_v1": "panvizplus_v1.yaml",
    "permissive_screening_2026_1": "permissive_screening_2026_1.yaml",
    "prolif_style_2026_1": "prolif_style_2026_1.yaml",
}


def load_ruleset(
    path: str | Path | None = None,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Load and validate one versioned rule profile."""
    if path is not None and profile is not None:
        raise ValueError("Specify either path or profile, not both.")

    if path is None:
        selected = profile or "panvizplus_v1"
        filename = _RULESET_FILES.get(selected)
        if filename is None:
            raise KeyError(
                f"Unknown panVizPlus rule profile: {selected}. "
                f"Available: {', '.join(sorted(_RULESET_FILES))}"
            )
        resource = files("panvizplus.rules").joinpath(filename)
        with resource.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

    _validate_ruleset(data)
    return data


def list_rulesets() -> list[dict[str, Any]]:
    """Return metadata for built-in profiles in stable display order."""
    out = []
    for profile_id in _RULESET_FILES:
        data = load_ruleset(profile=profile_id)
        meta = dict(data["metadata"])
        meta["profile_id"] = profile_id
        out.append(meta)
    return out


def _validate_ruleset(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Ruleset root must be a mapping.")

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Ruleset must contain a metadata mapping.")

    missing = _REQUIRED_METADATA.difference(metadata)
    if missing:
        raise ValueError(f"Ruleset metadata missing: {', '.join(sorted(missing))}")

    interactions = data.get("interactions")
    if not isinstance(interactions, dict) or not interactions:
        raise ValueError("Ruleset must contain at least one interaction definition.")
