"""One JSON record per run, as required by docs/experiment_protocol.md."""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "torch", "pennylane", "scikit-learn"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover
            out[pkg] = None
    return out


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**record, "versions": package_versions()}, indent=2))
