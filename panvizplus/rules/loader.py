from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_METADATA = {"id", "version", "status", "exact_biovia_reproduction"}


def load_ruleset(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        resource = files("panvizplus.rules").joinpath("panvizplus_v1.yaml")
        with resource.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

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

    return data
