"""
scripts/run_cheque.py
---------------------
Demo script: process a cheque through the full LangGraph workflow.

Auto-approve path (confidence >= 0.80):
  python scripts/run_cheque.py

HITL path (low confidence):
  python scripts/run_cheque.py --hitl

Set APP_ENV to control which prompt version is used:
  APP_ENV=staging python scripts/run_cheque.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langgraph.types import Command


def run_auto_approve():
    """High-confidence cheque — auto-approves without human review."""
    from src.agents.workflow import build_workflow

    app = build_workflow()
    config = {"configurable": {"thread_id": "demo-auto-001"}}

    print("\n🏦 Banking AI — Cheque Processing Demo (Auto-Approve)")
    print("─" * 54)

    result = app.invoke(
        {
            "cheque_id": "CHQ-AUTO-001",
            "cheque_text": (
                "Pay to the order of Jane Smith *** $2,500.00 "
                "Two Thousand Five Hundred Dollars  Date: 2024-01-15  "
                "Account: 9876543210  MICR: 000123|001|9876543210"
            ),
            "audit_log": [],
        },
        config=config,
    )

    print("Audit trail:")
    for entry in result["audit_log"]:
        print(f"  [{entry['ts']}]  {entry['event']}")


def run_hitl():
    """Low-confidence cheque — pauses for human review then resumes."""
    from src.agents.workflow import build_workflow

    app = build_workflow()
    config = {"configurable": {"thread_id": "demo-hitl-001"}}

    print("\n🏦 Banking AI — Cheque Processing Demo (HITL)")
    print("─" * 54)

    # Phase 1: start the graph — it will pause at the HITL node
    print("\nPhase 1: Starting graph (expect HITL interrupt)...")
    result = app.invoke(
        {
            "cheque_id": "CHQ-HITL-001",
            "cheque_text": (
                # Deliberately vague — missing date, ambiguous amount
                "Pay to J Smith  Five Hundred  $500"
            ),
            "audit_log": [],
        },
        config=config,
    )

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n⏸  Graph paused — awaiting human review")
        print(f"   Cheque ID : {payload['cheque_id']}")
        print(f"   Confidence: {payload['confidence']:.2f}")
        print(f"   Errors    : {payload['validation_errors']}")
        print(f"   Extracted : {payload['extracted_fields']}")

        # Simulate reviewer decision
        print("\nPhase 2: Reviewer approves with corrections...")
        final = app.invoke(
            Command(resume={
                "decision":    "approve",
                "corrections": {
                    "payee_name":     "Jane Smith",
                    "cheque_date":    "2024-01-15",
                    "amount_numeric": 500.0,
                    "amount_words":   "Five Hundred Dollars",
                },
            }),
            config=config,
        )

        print("\nAudit trail:")
        for entry in final["audit_log"]:
            print(f"  [{entry['ts']}]  {entry['event']}")
    else:
        print("Graph completed without HITL (check confidence threshold).")
        print(f"Status: {result.get('status')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banking AI cheque demo")
    parser.add_argument("--hitl", action="store_true",
                        help="Run the HITL (human-in-the-loop) demo path")
    args = parser.parse_args()

    if args.hitl:
        run_hitl()
    else:
        run_auto_approve()
