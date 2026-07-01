@echo off
REM fix_langsmith_plugin.bat
REM -------------------------
REM ONE-TIME SETUP: Removes langsmith's pytest plugin entry point so
REM pytest NEVER auto-loads it, regardless of how you invoke pytest.
REM
REM This is the permanent fix for:
REM   ImportError: DLL load failed while importing _uuid_utils:
REM   An Application Control policy has blocked this file.
REM
REM After running this once, plain "pytest tests/ -v" works with no
REM env vars, no wrapper scripts, no flags required.
REM
REM Usage (run once, from the project root):
REM   scripts\fix_langsmith_plugin.bat

python scripts\fix_langsmith_plugin.py
