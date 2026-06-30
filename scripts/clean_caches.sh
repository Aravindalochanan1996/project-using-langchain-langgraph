#!/usr/bin/env bash
# clean_caches.sh
# ----------------
# Removes all Python bytecode caches and pytest caches.
# Run this if you see strange "unknown location" or partially-imported
# module errors — Python can leave stale __pycache__ entries that
# reference broken import states from earlier failed runs.
#
# Usage:
#   bash scripts/clean_caches.sh

echo "Cleaning __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "Cleaning .pytest_cache..."
rm -rf .pytest_cache

echo "Done. Run pytest tests/ -v again."
