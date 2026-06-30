"""
conftest.py
-----------
Pytest auto-discovers and executes this file before collecting any
tests in the project. We use that to install the pure-Python uuid_utils
shim (see src/_uuid_utils_shim.py) BEFORE any test module gets a chance
to `import langgraph` or `from langchain_core...`, which would otherwise
try to load the real (Application-Control-blocked) compiled uuid_utils
extension on restricted corporate Windows machines.

This file does nothing on machines where uuid_utils works normally
(Mac/Linux, or Windows machines without restrictive policies) — it just
pre-empts the slower/blocked path with an equally correct pure-Python one.
"""

import sys
from pathlib import Path

# Ensure src/ is importable even if pytest's rootdir insertion order varies
sys.path.insert(0, str(Path(__file__).parent))

from src._uuid_utils_shim import install as _install_uuid_shim  # noqa: E402

_install_uuid_shim()
