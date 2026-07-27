"""Shared helpers for locally-built nanobind extension backends."""

from __future__ import annotations

import sys
from pathlib import Path


def import_local_extension(module_name: str, this_file: str):
    """Import *module_name*, falling back to the local build/ directory's
    compiled nanobind extension if the pip-installed package isn't found.
    """
    try:
        return __import__(module_name)
    except ImportError:
        pkg = Path(this_file).resolve().parent
        root = pkg
        while root.name not in ("solvers", "dev"):
            root = root.parent
        build = root / "build"
        for so in build.glob(f"{module_name}.*.so"):
            if str(build) not in sys.path:
                sys.path.insert(0, str(build))
        return __import__(module_name)
