"""
LangGraph nodes for the Banking AI cheque-processing pipeline.

Node contract:
  - Receives the full ChequeState dict
  - Returns a dict of ONLY the keys it updates
  - LangGraph merges the result back into state automatically
"""

import json
import os
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel, field_validator

from src.registry.prompt_registry import PromptRegistry

registry = PromptRegistry(prompts_dir="prompts")


# ── Pydantic output schema ────────────────────────────────────────────────────

class ChequeFields(BaseModel):
    amount_numeric: float | None = None
    amount_words: str | None = None
    payee_name: str | None = None
    cheque_date: str | None = None
    account_number: str | None = None
    bank_name: str | None = None
    micr_line: str | None = None
    confidence_note: str | None = None

    @field_validator("amount_numeric")
    @classmethod
    def must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("amount_numeric must be positive")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(state: dict, event: str) -> list:
    """Append a timestamped event to the audit log."""
    log = list(state.get("audit_log", []))
    log.append({"ts": datetime.now(timezone.utc).isoformat(), "event": event})
    return log


def _confidence(fields: ChequeFields) -> float:
    """
    Heuristic confidence score: fraction of critical fields present.
    Production systems would use the model's token-level log-probs instead.
    """
    critical = [fields.amount_numeric, fields.amount_words,
                fields.payee_name, fields.cheque_date]
    return sum(1 for f in critical if f is not None) / len(critical)


# ── Node 1: Extract ───────────────────────────────────────────────────────────

def extract_fields_node(state: dict) -> dict:
    """
    Call the versioned LLM prompt to extract structured fields
    from the raw cheque text.
    """
    env = os.getenv("APP_ENV", "production")
    prompt = registry.get("cheque_extraction", env=env)

    # Build message list, prepending any few-shot examples from the YAML
    messages = [{"role": "system", "content": prompt.system}]
    for ex in prompt.few_shot_examples:
        messages.append({"role": "user",      "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": state["cheque_text"]})

    llm = ChatOpenAI(
        model=prompt.model,
        temperature=prompt.temperature,
        max_tokens=prompt.max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    try:
        response = llm.invoke(messages)
        parsed = ChequeFields(**json.loads(response.content))
        confidence = _confidence(parsed)
        return {
            "extracted_fields": parsed.model_dump(),
            "confidence": confidence,
            "prompt_version": prompt.version,
            "audit_log": _log(
                state,
                f"Extracted fields | confidence={confidence:.2f} | "
                f"prompt_version={prompt.version}"
            ),
        }
    except Exception as exc:
        return {
            "extracted_fields": {},
            "confidence": 0.0,
            "prompt_version": prompt.version,
            "status": "error",
            "audit_log": _log(state, f"Extraction error: {exc}"),
        }


# ── Node 2: Validate ──────────────────────────────────────────────────────────

def validate_fields_node(state: dict) -> dict:
    """
    Apply business rules to extracted fields:
      - All critical fields must be present
      - Numeric amount and words amount must be consistent
    """
    f = state.get("extracted_fields", {})
    errors = []
    amount_mismatch = False

    if not f.get("payee_name"):
        errors.append("Payee name is missing")
    if f.get("amount_numeric") is None:
        errors.append("Numeric amount is missing")
    if not f.get("amount_words"):
        errors.append("Amount in words is missing")
    if not f.get("cheque_date"):
        errors.append("Cheque date is missing")

    # Cross-check numeric vs words using magnitude keywords.
    # Strategy: derive the order-of-magnitude keyword from the numeric amount
    # (e.g. 1500 → "thousand", 10000 → "thousand", 1500000 → "million")
    # and check it appears in the words string.
    # This avoids false positives from purely substring matching digits in words.
    if f.get("amount_numeric") and f.get("amount_words"):
        numeric = float(f["amount_numeric"])
        words_lower = f["amount_words"].lower()

        magnitude_ok = True
        if numeric >= 1_000_000 and "million" not in words_lower:
            magnitude_ok = False
        elif 1_000 <= numeric < 1_000_000 and "thousand" not in words_lower:
            magnitude_ok = False
        elif numeric < 1_000 and any(
            kw in words_lower
            for kw in ("thousand", "million", "billion")
        ):
            magnitude_ok = False

        if not magnitude_ok:
            amount_mismatch = True
            errors.append(
                f"Amount mismatch: numeric={f['amount_numeric']} "
                f"vs words='{f['amount_words']}'"
            )

    needs_review = bool(errors) or state.get("confidence", 1.0) < 0.80

    return {
        "validation_errors": errors,
        "amount_mismatch": amount_mismatch,
        "needs_human_review": needs_review,
        "audit_log": _log(
            state,
            f"Validated | errors={len(errors)} | needs_review={needs_review}"
        ),
    }


# ── Node 3a: HITL Review ──────────────────────────────────────────────────────

def hitl_review_node(state: dict) -> dict:
    """
    Human-in-the-Loop pause node.

    interrupt() serialises the full graph state to the checkpointer
    and suspends execution. The calling application receives the
    interrupt payload and shows it to a human reviewer.

    Execution resumes when the app calls graph.invoke() again with
    the SAME thread_id and Command(resume={decision, corrections}).
    """
    human_input = interrupt({
        "cheque_id":        state["cheque_id"],
        "extracted_fields": state["extracted_fields"],
        "confidence":       state["confidence"],
        "validation_errors":state["validation_errors"],
        "amount_mismatch":  state["amount_mismatch"],
        "instruction": (
            "Please review the extracted fields. "
            "Return {'decision': 'approve'|'reject', "
            "'corrections': {field: value, ...}}"
        ),
    })

    return {
        "human_decision":   human_input.get("decision"),
        "corrected_fields": human_input.get("corrections"),
        "audit_log": _log(
            state,
            f"HITL decision: {human_input.get('decision')}"
        ),
    }


# ── Node 3b: Auto-Approve ─────────────────────────────────────────────────────

def auto_approve_node(state: dict) -> dict:
    """High-confidence, error-free path — no human reviewer needed."""
    return {
        "human_decision":   "approve",
        "corrected_fields": None,
        "audit_log": _log(state, "Auto-approved (confidence ≥ 0.80, no errors)"),
    }


# ── Node 4: Finalise ──────────────────────────────────────────────────────────

def finalise_node(state: dict) -> dict:
    """
    Merge extracted fields with any reviewer corrections,
    set the final status, and (in production) write to core banking.
    """
    base = dict(state.get("extracted_fields", {}))
    base.update(state.get("corrected_fields") or {})  # corrections win

    decision = state.get("human_decision", "approve")
    status = "approved" if decision == "approve" else "rejected"

    print(f"\n{'='*52}")
    print(f"  Cheque {state['cheque_id']}  →  {status.upper()}")
    if status == "approved":
        print(f"  Payee  : {base.get('payee_name')}")
        print(f"  Amount : {base.get('amount_numeric')} ({base.get('amount_words')})")
        print(f"  Date   : {base.get('cheque_date')}")
    print(f"{'='*52}\n")

    return {
        "final_fields": base,
        "status": status,
        "audit_log": _log(state, f"Finalised | status={status}"),
    }
