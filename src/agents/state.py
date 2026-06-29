"""
ChequeState — shared state object for the LangGraph workflow.

Every node reads from and writes to this TypedDict.
LangGraph merges each node's returned dict back into the full state,
so nodes only need to return the keys they actually change.
"""

from typing import Optional, TypedDict


class ChequeState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    cheque_id: str          # unique ID for this processing run
    cheque_text: str        # OCR text / description of the cheque image

    # ── Extraction ─────────────────────────────────────────────────────────
    extracted_fields: dict  # structured JSON returned by the LLM
    confidence: float       # 0.0–1.0 based on how many critical fields found
    prompt_version: str     # version of the prompt used (e.g. "1.0.0")

    # ── Validation ─────────────────────────────────────────────────────────
    validation_errors: list # list of human-readable error strings
    amount_mismatch: bool   # True when numeric amount ≠ words amount

    # ── HITL ───────────────────────────────────────────────────────────────
    needs_human_review: bool
    human_decision: Optional[str]     # "approve" | "reject"
    corrected_fields: Optional[dict]  # fields the reviewer fixed

    # ── Output ─────────────────────────────────────────────────────────────
    final_fields: dict      # merged extracted + corrections → core banking
    status: str             # "approved" | "rejected" | "error"
    audit_log: list         # chronological list of {ts, event} dicts
