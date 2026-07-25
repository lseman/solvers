"""optBlocks sparse solvers package."""

import importlib
import sys
from pathlib import Path

# Installed extensions are package siblings. Development builds place them in
# build/solvers, so locate the repository root once for that fallback.
_pkg_dir = Path(__file__).resolve().parent
_project_root = _pkg_dir.parent
while _project_root.name != "solvers" and _project_root.name != "dev":
    _project_root = _project_root.parent

_build_dir = _project_root / "build" / "solvers"


def _load_extension(module_name, attribute):
    try:
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attribute, None)
    except ImportError:
        extension = next(_build_dir.glob(f"{module_name}.*.so"), None)
        if extension is None:
            return None
        build_path = str(_build_dir)
        if build_path not in sys.path:
            sys.path.insert(0, build_path)

    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute, None)
    except ImportError:
        return None


SupernodalLDLT = _load_extension("supernodal", "SupernodalLDLT")
identify_supernodes = _load_extension("supernodes", "identify_supernodes")
__all__ = ["SupernodalLDLT", "identify_supernodes"]
del _build_dir, _pkg_dir, _project_root, _load_extension
