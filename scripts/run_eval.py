"""
scripts/run_eval.py
-------------------
Run the prompt eval suite and optionally promote on pass.

Examples:
  python scripts/run_eval.py --env dev
  python scripts/run_eval.py --env dev --promote --to staging
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Install pure-Python uuid_utils shim BEFORE any langgraph/langchain_core
# import, so this works even on machines where the compiled uuid_utils
# extension is blocked by Application Control policies. See
# src/_uuid_utils_shim.py for details.
from src._uuid_utils_shim import install as _install_uuid_shim
_install_uuid_shim()

from src.evaluation.eval_suite import print_report, promote_if_passing, run_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt eval suite")
    parser.add_argument("--env",     default="dev",     help="Source environment")
    parser.add_argument("--to",      default="staging", help="Target environment for promotion")
    parser.add_argument("--promote", action="store_true", help="Auto-promote if passing")
    parser.add_argument("--prompt",  default="cheque_extraction", help="Prompt name")
    args = parser.parse_args()

    if args.promote:
        promote_if_passing(args.prompt, from_env=args.env, to_env=args.to)
    else:
        result = run_eval(args.prompt, env=args.env)
        print_report(result, args.prompt, args.env)
