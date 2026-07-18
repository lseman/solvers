"""
Pytest conftest — ensure build/ is on sys.path so .so modules are importable.
"""

import sys
from pathlib import Path

# Add project root's build/ so qdldl, ldlt, ldlt_bk, solvers.* etc. resolve at import time.
_project_root = Path(__file__).resolve().parent.parent.parent  # → .../solvers/
_build = _project_root / "build"
if str(_build) not in sys.path:
    sys.path.insert(0, str(_build))
