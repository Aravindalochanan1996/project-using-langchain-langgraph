"""
scripts/fix_langsmith_plugin.py
--------------------------------
ONE-TIME SETUP: Permanently neutralizes langsmith's pytest plugin so it
is never auto-discovered by pytest via setuptools entry points.

WHY THIS IS NEEDED
------------------
On Windows machines with Application Control / AppLocker policies,
langsmith's pytest plugin crashes pytest before any test runs:

  ImportError: DLL load failed while importing _uuid_utils:
  An Application Control policy has blocked this file.

The plugin is auto-loaded by pytest via setuptools entry points
(load_setuptools_entrypoints) during pytest's own startup phase —
BEFORE conftest.py, BEFORE pytest.ini addopts, BEFORE any Python code
in this project executes. There is no purely in-Python way to intercept
this without setting an environment variable (PYTEST_DISABLE_PLUGIN_AUTOLOAD).

WHAT THIS SCRIPT DOES
---------------------
It finds the `entry_points.txt` file inside langsmith's .dist-info
directory (in your venv's site-packages) and removes the [pytest11]
section that registers the plugin. The langsmith package itself remains
fully installed and importable — only the pytest plugin registration
is removed. This is safe and reversible (reinstalling langsmith restores
it, but you'd need to run this script again).

USAGE
-----
Run once from the project root:
  python scripts/fix_langsmith_plugin.py

You only need to run this once per venv. After this, plain
`pytest tests/ -v` works with no env vars or flags.
"""

import sys
from pathlib import Path


def find_langsmith_dist_info() -> Path | None:
    """Find the langsmith .dist-info directory in the active environment."""
    try:
        import importlib.metadata
        dist = importlib.metadata.distribution("langsmith")
        # dist._path is the .dist-info directory itself
        dist_info = Path(str(dist._path))
        if dist_info.is_dir():
            return dist_info
    except importlib.metadata.PackageNotFoundError:
        pass

    # Fallback: scan site-packages directly
    for path in sys.path:
        p = Path(path)
        if p.is_dir():
            for d in p.glob("langsmith-*.dist-info"):
                return d
    return None


def fix() -> None:
    dist_info = find_langsmith_dist_info()
    if not dist_info:
        print("❌  Could not find langsmith installation. Is it installed?")
        print("    Run: pip install langsmith")
        sys.exit(1)

    print(f"✅  Found langsmith dist-info: {dist_info}")

    ep_file = dist_info / "entry_points.txt"
    if not ep_file.exists():
        print("✅  No entry_points.txt found — plugin already not registered.")
        return

    original = ep_file.read_text(encoding="utf-8")
    print(f"\nCurrent entry_points.txt:\n{original.strip()}\n")

    if "[pytest11]" not in original:
        print("✅  No [pytest11] section found — plugin already not registered.")
        return

    # Remove the [pytest11] section and everything under it
    lines = original.splitlines(keepends=True)
    new_lines = []
    skip = False
    for line in lines:
        if line.strip() == "[pytest11]":
            skip = True          # start skipping this section
            continue
        if skip and line.strip().startswith("["):
            skip = False         # next section started — stop skipping
        if not skip:
            new_lines.append(line)

    new_content = "".join(new_lines).rstrip() + "\n"

    # Back up the original
    backup = ep_file.with_suffix(".txt.bak")
    backup.write_text(original, encoding="utf-8")
    print(f"📄  Backed up original to: {backup}")

    ep_file.write_text(new_content, encoding="utf-8")
    print(f"\nUpdated entry_points.txt:\n{new_content.strip()}\n")
    print("✅  Done! langsmith's pytest plugin is now permanently deregistered.")
    print("    You can now run: pytest tests/ -v")
    print("\n    To undo: copy the .bak file back over entry_points.txt")


if __name__ == "__main__":
    fix()
