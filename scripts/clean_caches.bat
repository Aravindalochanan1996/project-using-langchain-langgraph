@echo off
REM clean_caches.bat
REM -----------------
REM Removes all Python bytecode caches and pytest caches.
REM Run this if you see strange "unknown location" or partially-imported
REM module errors after switching between different fix attempts — Python
REM can leave stale __pycache__ entries that reference broken import states.
REM
REM Usage:
REM   scripts\clean_caches.bat

echo Cleaning __pycache__ directories...
for /d /r %%i in (__pycache__) do (
    if exist "%%i" rd /s /q "%%i"
)

echo Cleaning .pytest_cache...
if exist .pytest_cache rd /s /q .pytest_cache

echo Done. Run pytest tests/ -v again.
