"""
_uuid_utils_shim.py
--------------------
Pure-Python replacement for the `uuid_utils` package.

WHY THIS EXISTS:
  langchain_core (and therefore langgraph and langsmith) depend on
  `uuid_utils`, a Rust-compiled package, purely to generate UUIDv7
  identifiers slightly faster than pure Python. On machines with strict
  corporate Application Control / AppLocker policies, the compiled
  extension (_uuid_utils.pyd) gets blocked outright:

      ImportError: DLL load failed while importing _uuid_utils:
      An Application Control policy has blocked this file.

  This is purely a performance dependency, not a functional one — any
  RFC-4122-compliant UUIDv7 generator works identically from
  langchain_core's perspective. This module provides one using only
  Python's stdlib `uuid` and `secrets` modules, with zero compiled code,
  so it can never be blocked by Application Control policies.

HOW IT'S ACTIVATED:
  Call `install()` BEFORE importing langgraph, langchain_core, or
  langsmith anywhere in the process. This pre-populates sys.modules
  with the shim so the real (blocked) uuid_utils is never imported.

  This is already wired in for you in:
    - conftest.py           (for pytest test runs)
    - scripts/run_cheque.py (for the demo CLI)
    - scripts/run_eval.py   (for the eval CLI)

  You do not need to call this manually unless you add a new entry
  point script.
"""

import secrets
import sys
import time
import types
import uuid as _stdlib_uuid


def _uuid7() -> _stdlib_uuid.UUID:
    """
    Generate an RFC-9562-compliant UUIDv7: a 48-bit millisecond Unix
    timestamp followed by random bits, with version/variant fields set.
    Pure Python — no compiled extensions involved.
    """
    ts_ms = int(time.time() * 1000)
    ts_bytes = ts_ms.to_bytes(6, "big")
    rand_bytes = secrets.token_bytes(10)

    b = bytearray(ts_bytes + rand_bytes)
    b[6] = (b[6] & 0x0F) | 0x70  # version 7
    b[8] = (b[8] & 0x3F) | 0x80  # RFC 4122 variant

    return _stdlib_uuid.UUID(bytes=bytes(b))


def install() -> None:
    """
    Register a pure-Python `uuid_utils` module (and `uuid_utils.compat`)
    in sys.modules, so any subsequent `import uuid_utils` anywhere in
    the process — by our code or by langchain_core/langsmith — receives
    this shim instead of trying to load the real compiled package.

    Safe to call multiple times; only installs once.
    """
    if "uuid_utils" in sys.modules and getattr(
        sys.modules["uuid_utils"], "_is_pure_python_shim", False
    ):
        return  # already installed

    shim = types.ModuleType("uuid_utils")
    shim._is_pure_python_shim = True
    shim.uuid7 = _uuid7
    shim.uuid1 = _stdlib_uuid.uuid1
    shim.uuid3 = _stdlib_uuid.uuid3
    shim.uuid4 = _stdlib_uuid.uuid4
    shim.uuid5 = _stdlib_uuid.uuid5
    shim.UUID = _stdlib_uuid.UUID
    shim.NAMESPACE_DNS = _stdlib_uuid.NAMESPACE_DNS
    shim.NAMESPACE_URL = _stdlib_uuid.NAMESPACE_URL
    shim.NAMESPACE_OID = _stdlib_uuid.NAMESPACE_OID
    shim.NAMESPACE_X500 = _stdlib_uuid.NAMESPACE_X500
    shim.NIL = _stdlib_uuid.UUID(int=0)
    shim.SafeUUID = _stdlib_uuid.SafeUUID

    sys.modules["uuid_utils"] = shim

    # langsmith imports `from uuid_utils.compat import uuid7` separately
    compat = types.ModuleType("uuid_utils.compat")
    compat.uuid7 = _uuid7
    sys.modules["uuid_utils.compat"] = compat
