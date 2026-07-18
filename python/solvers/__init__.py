"""optBlocks sparse solvers package."""

import importlib
import sys
from pathlib import Path

# ── Locate the build .so ──────────────────────────────────────────────────────
# The build puts supernodal.cpython-*.so into build/solvers/.
# The source package lives in python/solvers/.  We need to find the .so
# relative to this __init__.py.
_pkg_dir = Path(__file__).resolve().parent  # python/solvers/
_project_root = _pkg_dir.parent  # python/
# Walk up to find the project root (solvers/)
while _project_root.name != "solvers" and _project_root.name != "dev":
    _project_root = _project_root.parent

_build_dir = _project_root / "build" / "solvers"
_so = next(_build_dir.glob("supernodal.*.so"), None)

if _so is not None:
    # The .so exports a module named "supernodal" (output_name in CMake).
    # Add the solvers/ build directory to sys.path so we can import it.
    _build_dir_str = str(_so.parent)
    if _build_dir_str not in sys.path:
        sys.path.insert(0, _build_dir_str)
    try:
        _mod = importlib.import_module("supernodal")
        SupernodalLDLT = getattr(_mod, "SupernodalLDLT", None)
        del _mod
    except Exception:
        SupernodalLDLT = None
    del _so, _build_dir_str
else:
    SupernodalLDLT = None

__all__ = ["SupernodalLDLT"]
del _build_dir, _build_dir_str, _pkg_dir, _project_root
