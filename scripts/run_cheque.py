"""
scripts/run_cheque.py
---------------------
Demo script: process a cheque through the full LangGraph workflow.

Two modes:
  --mock  (default) Uses mock LLM responses — works with NO API key.
                    Great for demos, interviews, and local testing.
  --live            Uses real OpenAI GPT-4o. Requires OPENAI_API_KEY in .env

Usage:
  python scripts/run_cheque.py              # mock mode, auto-approve path
  python scripts/run_cheque.py --hitl       # mock mode, HITL path
  python scripts/run_cheque.py --live       # live LLM, auto-approve
  python scripts/run_cheque.py --live --hitl  # live LLM, HITL path

APP_ENV controls which prompt version is loaded (dev/staging/production):
  APP_ENV=staging python scripts/run_cheque.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Install pure-Python uuid_utils shim BEFORE any langgraph/langchain_core
# import so this works on machines with Application Control policies.
from src._uuid_utils_shim import install as _install_uuid_shim
_install_uuid_shim()

from langgraph.types import Command


# ── Mock LLM response (used when --live flag is NOT passed) ──────────────────

MOCK_RESPONSES = {
    "high_confidence": {
        "amount_numeric":  2500.0,
        "amount_words":    "Two Thousand Five Hundred Dollars",
        "payee_name":      "Jane Smith",
        "cheque_date":     "2024-01-15",
        "account_number":  "9876543210",
        "bank_name":       "ADCB",
        "micr_line":       "000123|001|9876543210",
        "confidence_note": "All fields clearly legible",
    },
    "low_confidence": {
        "amount_numeric":  500.0,
        "amount_words":    None,          # missing — triggers validation error
        "payee_name":      None,          # missing — triggers HITL
        "cheque_date":     None,          # missing — triggers HITL
        "account_number":  None,
        "bank_name":       None,
        "micr_line":       None,
        "confidence_note": "Handwriting unclear — multiple fields unreadable",
    },
}


def _patch_extract_node_with_mock(mock_key: str):
    """
    Replace the real LLM extract node with a mock that returns
    pre-defined fields instantly, no API key required.
    """
    import src.agents.nodes as nodes_mod
    import src.agents.workflow as wf_mod

    fields = MOCK_RESPONSES[mock_key]
    critical = [fields["amount_numeric"], fields["amount_words"],
                fields["payee_name"], fields["cheque_date"]]
    confidence = sum(1 for f in critical if f is not None) / len(critical)

    def mock_extract(state: dict) -> dict:
        print(f"  [Mock LLM] Returning pre-set fields (confidence={confidence:.2f})")
        return {
            "extracted_fields": fields,
            "confidence":       confidence,
            "prompt_version":   "1.0.0-mock",
            "audit_log": list(state.get("audit_log", [])) + [{
                "ts":    "2024-01-15T08:00:00+00:00",
                "event": f"Mock extraction | confidence={confidence:.2f}",
            }],
        }

    nodes_mod.extract_fields_node = mock_extract
    wf_mod.extract_fields_node    = mock_extract
    return nodes_mod, wf_mod  # keep reference so GC doesn't clean up


def _restore_extract_node(nodes_mod, wf_mod, original_fn):
    nodes_mod.extract_fields_node = original_fn
    wf_mod.extract_fields_node    = original_fn


# ── Sample cheque data ────────────────────────────────────────────────────────

SAMPLE_CHEQUES = {
    "high_confidence": {
        "id":   "CHQ-AUTO-001",
        "text": (
            "Pay to the order of Jane Smith *** $2,500.00\n"
            "Two Thousand Five Hundred Dollars\n"
            "Date: 2024-01-15\n"
            "Account: 9876543210\n"
            "MICR: 000123|001|9876543210"
        ),
    },
    "low_confidence": {
        "id":   "CHQ-HITL-001",
        "text": (
            "Pay to J S***  F*** H*****  $500\n"
            "(handwriting unclear — multiple fields illegible)"
        ),
    },
}


# ── Runners ───────────────────────────────────────────────────────────────────

def run_auto_approve(use_mock: bool = True):
    """High-confidence cheque — should auto-approve without human review."""
    from src.agents.workflow import build_workflow
    import src.agents.nodes as nodes_mod

    cheque = SAMPLE_CHEQUES["high_confidence"]

    print(f"\n{'─'*54}")
    print("🏦  Banking AI — Cheque Processing Demo")
    print(f"{'─'*54}")
    print(f"  Mode    : {'Mock LLM (no API key needed)' if use_mock else 'Live OpenAI GPT-4o'}")
    print(f"  Path    : Auto-Approve")
    print(f"  Cheque  : {cheque['id']}")
    print(f"\n  Cheque text:\n")
    for line in cheque["text"].splitlines():
        print(f"    {line}")
    print()

    original_fn = nodes_mod.extract_fields_node
    mods = None

    try:
        if use_mock:
            mods = _patch_extract_node_with_mock("high_confidence")

        app = build_workflow()
        result = app.invoke(
            {
                "cheque_id":          cheque["id"],
                "cheque_text":        cheque["text"],
                "audit_log":          [],
                "extracted_fields":   {},
                "confidence":         0.0,
                "prompt_version":     "",
                "validation_errors":  [],
                "amount_mismatch":    False,
                "needs_human_review": False,
                "human_decision":     None,
                "corrected_fields":   None,
                "final_fields":       {},
                "status":             "",
            },
            config={"configurable": {"thread_id": cheque["id"]}},
        )
    finally:
        if mods:
            _restore_extract_node(mods[0], mods[1], original_fn)

    print("\n📋  Audit trail:")
    for entry in result.get("audit_log", []):
        print(f"    [{entry['ts']}]  {entry['event']}")


def run_hitl(use_mock: bool = True):
    """Low-confidence cheque — HITL pause, reviewer corrects, then resumes."""
    from src.agents.workflow import build_workflow
    import src.agents.nodes as nodes_mod

    cheque = SAMPLE_CHEQUES["low_confidence"]

    print(f"\n{'─'*54}")
    print("🏦  Banking AI — Cheque Processing Demo")
    print(f"{'─'*54}")
    print(f"  Mode    : {'Mock LLM (no API key needed)' if use_mock else 'Live OpenAI GPT-4o'}")
    print(f"  Path    : Human-in-the-Loop (HITL)")
    print(f"  Cheque  : {cheque['id']}")
    print(f"\n  Cheque text:\n")
    for line in cheque["text"].splitlines():
        print(f"    {line}")
    print()

    original_fn = nodes_mod.extract_fields_node
    mods = None

    try:
        if use_mock:
            mods = _patch_extract_node_with_mock("low_confidence")

        app = build_workflow()
        config = {"configurable": {"thread_id": cheque["id"]}}

        # ── Phase 1: Start graph — will pause at HITL node ──────────────────
        print("Phase 1: Running extract → validate → HITL pause...")
        result = app.invoke(
            {
                "cheque_id":          cheque["id"],
                "cheque_text":        cheque["text"],
                "audit_log":          [],
                "extracted_fields":   {},
                "confidence":         0.0,
                "prompt_version":     "",
                "validation_errors":  [],
                "amount_mismatch":    False,
                "needs_human_review": False,
                "human_decision":     None,
                "corrected_fields":   None,
                "final_fields":       {},
                "status":             "",
            },
            config=config,
        )

        interrupt_data = (result.get("__interrupt__") or [None])[0]
        if interrupt_data:
            payload = interrupt_data.value if hasattr(interrupt_data, "value") else interrupt_data

            print(f"\n⏸   Graph PAUSED — awaiting human review")
            print(f"    Cheque ID   : {payload.get('cheque_id')}")
            print(f"    Confidence  : {payload.get('confidence', 0):.0%}")
            print(f"    Errors      : {payload.get('validation_errors', [])}")
            print(f"\n    Extracted fields:")
            for k, v in (payload.get("extracted_fields") or {}).items():
                print(f"      {k:20}: {v}")

            # ── Phase 2: Simulate reviewer decision ──────────────────────────
            print(f"\n{'─'*54}")
            print("👤  Simulated reviewer decision:")
            corrections = {
                "payee_name":     "Jane Smith",
                "cheque_date":    "2024-01-15",
                "amount_words":   "Five Hundred Dollars",
                "account_number": "1234567890",
            }
            print(f"    Decision    : APPROVE")
            print(f"    Corrections :")
            for k, v in corrections.items():
                print(f"      {k:20}: {v}")

            print(f"\nPhase 2: Resuming graph with reviewer decision...")
            final = app.invoke(
                Command(resume={
                    "decision":    "approve",
                    "corrections": corrections,
                }),
                config=config,
            )

            print("\n📋  Full audit trail:")
            for entry in final.get("audit_log", []):
                print(f"    [{entry['ts']}]  {entry['event']}")
        else:
            print("ℹ️   Graph completed without HITL pause.")
            print(f"    Status: {result.get('status')}")
            print("    (Confidence may have been >= 0.80 — try --hitl with --live flag)")

    finally:
        if mods:
            _restore_extract_node(mods[0], mods[1], original_fn)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Banking AI cheque processing demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_cheque.py                # mock, auto-approve
  python scripts/run_cheque.py --hitl         # mock, HITL path
  python scripts/run_cheque.py --live         # real GPT-4o, needs OPENAI_API_KEY
  python scripts/run_cheque.py --live --hitl  # real GPT-4o, HITL path
        """,
    )
    parser.add_argument("--hitl", action="store_true",
                        help="Run the Human-in-the-Loop demo path")
    parser.add_argument("--live", action="store_true",
                        help="Use real OpenAI GPT-4o instead of mock responses")
    args = parser.parse_args()

    if args.live:
        # Load .env file if present
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅  Loaded .env file")
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("❌  OPENAI_API_KEY not set.")
            print("    Create a .env file with:  OPENAI_API_KEY=sk-...")
            print("    Or run without --live to use mock mode (no key needed).")
            sys.exit(1)

    if args.hitl:
        run_hitl(use_mock=not args.live)
    else:
        run_auto_approve(use_mock=not args.live)
