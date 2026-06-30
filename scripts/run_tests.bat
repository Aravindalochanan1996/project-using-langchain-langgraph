@echo off
REM run_tests.bat
REM --------------
REM Windows convenience wrapper that runs pytest with plugin
REM auto-loading disabled, which avoids the langsmith/uuid_utils
REM Application Control DLL-block crash on corporate machines.
REM
REM Usage:
REM   scripts\run_tests.bat

set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
pytest tests/ -v
