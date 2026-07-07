"""
Pytest conftest — ensure build/ is on sys.path so .so modules are importable.
"""

import sys
from pathlib import Path

# Add project root's build/ so qdldl, ldlt, ldlt_bk etc. resolve at import time.
_build = Path(__file__).resolve().parent.parent.parent / "build"
if str(_build) not in sys.path:
    sys.path.insert(0, str(_build))
