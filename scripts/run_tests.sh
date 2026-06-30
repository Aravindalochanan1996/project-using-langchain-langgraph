#!/usr/bin/env bash
# run_tests.sh
# ------------
# macOS/Linux convenience wrapper that runs pytest with plugin
# auto-loading disabled, avoiding the langsmith/uuid_utils crash
# seen on some restricted environments.
#
# Usage:
#   bash scripts/run_tests.sh

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
pytest tests/ -v
