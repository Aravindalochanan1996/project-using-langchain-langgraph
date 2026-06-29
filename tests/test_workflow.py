"""
Unit tests for the Banking AI cheque-processing pipeline.

Run with:  pytest tests/ -v

Tests use mocked LLM responses — no OpenAI key needed.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ── Prompt Registry tests ─────────────────────────────────────────────────────

class TestPromptRegistry:

    def test_get_production_prompt(self):
        from src.registry.prompt_registry import PromptRegistry
        reg = PromptRegistry(prompts_dir="prompts")
        prompt = reg.get("cheque_extraction", env="production")
        assert prompt.version == "1.0.0"
        assert "payee_name" in prompt.system
        assert prompt.temperature == 0.1

    def test_get_staging_prompt(self):
        from src.registry.prompt_registry import PromptRegistry
        reg = PromptRegistry(prompts_dir="prompts")
        prompt = reg.get("cheque_extraction", env="staging")
        assert prompt.version == "1.1.0"
        assert len(prompt.few_shot_examples) >= 1

    def test_get_dev_prompt(self):
        from src.registry.prompt_registry import PromptRegistry
        reg = PromptRegistry(prompts_dir="prompts")
        prompt = reg.get("cheque_extraction", env="dev")
        assert prompt.version == "1.2.0"
        assert len(prompt.few_shot_examples) >= 2

    def test_promote(self):
        from src.registry.prompt_registry import PromptRegistry
        reg = PromptRegistry(prompts_dir="prompts")
        reg.promote("cheque_extraction", "1.2.0", from_env="dev", to_env="staging")
        assert reg.active_version("cheque_extraction", "staging") == "1.2.0"

    def test_rollback(self):
        from src.registry.prompt_registry import PromptRegistry
        reg = PromptRegistry(prompts_dir="prompts")
        reg.rollback("cheque_extraction", env="production", to_version="1.0.0")
        assert reg.active_version("cheque_extraction", "production") == "1.0.0"

    def test_missing_prompt_raises(self):
        from src.registry.prompt_registry import PromptRegistry
        reg = PromptRegistry(prompts_dir="prompts")
        with pytest.raises(ValueError):
            reg.get("nonexistent_prompt", env="production")


# ── Validation node tests ─────────────────────────────────────────────────────

class TestValidationNode:

    def _base_state(self, fields: dict, confidence: float = 0.9):
        return {
            "cheque_id": "TEST-001",
            "extracted_fields": fields,
            "confidence": confidence,
            "audit_log": [],
        }

    def test_valid_fields_no_review_needed(self):
        from src.agents.nodes import validate_fields_node
        state = self._base_state({
            "payee_name":     "Jane Smith",
            "amount_numeric": 1000.0,
            "amount_words":   "One Thousand Dollars",
            "cheque_date":    "2024-01-15",
        })
        result = validate_fields_node(state)
        assert result["validation_errors"] == []
        assert result["needs_human_review"] is False

    def test_missing_payee_triggers_review(self):
        from src.agents.nodes import validate_fields_node
        state = self._base_state({
            "amount_numeric": 500.0,
            "amount_words":   "Five Hundred",
            "cheque_date":    "2024-01-15",
        })
        result = validate_fields_node(state)
        assert any("Payee" in e for e in result["validation_errors"])
        assert result["needs_human_review"] is True

    def test_low_confidence_triggers_review(self):
        from src.agents.nodes import validate_fields_node
        state = self._base_state(
            {
                "payee_name":     "Jane Smith",
                "amount_numeric": 1000.0,
                "amount_words":   "One Thousand",
                "cheque_date":    "2024-01-15",
            },
            confidence=0.5,
        )
        result = validate_fields_node(state)
        assert result["needs_human_review"] is True

    def test_missing_date_adds_error(self):
        from src.agents.nodes import validate_fields_node
        state = self._base_state({
            "payee_name":     "Jane Smith",
            "amount_numeric": 200.0,
            "amount_words":   "Two Hundred",
        })
        result = validate_fields_node(state)
        assert any("date" in e.lower() for e in result["validation_errors"])

    def test_amount_mismatch_detected(self):
        from src.agents.nodes import validate_fields_node
        # 1,000,000 is a million but words say "thousand" — clear mismatch
        state = self._base_state({
            "payee_name":     "Jane Smith",
            "amount_numeric": 1_000_000.0,
            "amount_words":   "One Thousand Dollars",
            "cheque_date":    "2024-01-15",
        })
        result = validate_fields_node(state)
        assert result["amount_mismatch"] is True
        assert result["needs_human_review"] is True


# ── Mock Azure tests ──────────────────────────────────────────────────────────

class TestMockAzure:

    def test_amount_extraction(self):
        from src.tools.mock_azure import analyze_cheque
        result = analyze_cheque("Pay to Jane Smith $2500.00")
        assert result.fields["amount_numeric"].content == "2500.00"
        assert result.fields["amount_numeric"].confidence > 0

    def test_micr_extraction(self):
        from src.tools.mock_azure import analyze_cheque
        result = analyze_cheque("MICR: 000123|001|9876543210")
        assert result.fields["micr_line"].content == "000123|001|9876543210"

    def test_missing_amount_returns_none(self):
        from src.tools.mock_azure import analyze_cheque
        result = analyze_cheque("Pay to John Doe dated 2024-01-15")
        assert result.fields["amount_numeric"].content is None
        assert result.fields["amount_numeric"].confidence == 0.0

    def test_date_extraction(self):
        from src.tools.mock_azure import analyze_cheque
        result = analyze_cheque("Cheque Date: 2024-03-20")
        assert result.fields["cheque_date"].content == "2024-03-20"


# ── Workflow integration test (LLM fully mocked via node patch) ───────────────

class TestWorkflow:

    GOOD_FIELDS = {
        "amount_numeric": 1500.0,
        "amount_words":   "One Thousand Five Hundred",
        "payee_name":     "Jane Smith",
        "cheque_date":    "2024-01-15",
        "account_number": None,
        "bank_name":      None,
        "micr_line":      None,
        "confidence_note":"Clear extraction",
    }

    def _full_initial_state(self, cheque_text: str, cheque_id: str = "CHQ-001"):
        return {
            "cheque_id":          cheque_id,
            "cheque_text":        cheque_text,
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
        }

    def test_auto_approve_path(self):
        """High-confidence cheque → auto-approved, no HITL."""
        good_fields = self.GOOD_FIELDS

        def mock_extract(state):
            return {
                "extracted_fields": good_fields,
                "confidence": 1.0,
                "prompt_version": "1.0.0",
                "audit_log": list(state.get("audit_log", [])) + [
                    {"ts": "2024-01-15T00:00:00+00:00", "event": "Mock extraction OK"}
                ],
            }

        with patch("src.agents.workflow.extract_fields_node", mock_extract):
            from src.agents.workflow import build_workflow
            app = build_workflow()
            result = app.invoke(
                self._full_initial_state(
                    "Pay to Jane Smith $1500 One Thousand Five Hundred Date: 2024-01-15"
                ),
                config={"configurable": {"thread_id": "test-auto-001"}},
            )

        assert result["status"] == "approved"
        assert result["human_decision"] == "approve"
        assert result["final_fields"]["payee_name"] == "Jane Smith"
        assert result["final_fields"]["amount_numeric"] == 1500.0
        assert len(result["audit_log"]) >= 3

    def test_rejected_path(self):
        """Simulate a HITL node that rejects the cheque."""
        good_fields = self.GOOD_FIELDS

        def mock_extract(state):
            return {
                "extracted_fields": good_fields,
                "confidence": 0.4,           # low → triggers HITL
                "prompt_version": "1.0.0",
                "audit_log": list(state.get("audit_log", [])) + [
                    {"ts": "2024-01-15T00:00:00+00:00", "event": "Mock extraction low-conf"}
                ],
            }

        def mock_hitl(state):
            # Simulate reviewer rejecting
            return {
                "human_decision":   "reject",
                "corrected_fields": None,
                "audit_log": list(state.get("audit_log", [])) + [
                    {"ts": "2024-01-15T00:00:01+00:00", "event": "Mock HITL: rejected"}
                ],
            }

        with patch("src.agents.workflow.extract_fields_node", mock_extract), \
             patch("src.agents.workflow.hitl_review_node", mock_hitl):
            from src.agents.workflow import build_workflow
            app = build_workflow()
            result = app.invoke(
                self._full_initial_state("Unclear cheque text", "CHQ-REJECT"),
                config={"configurable": {"thread_id": "test-reject-001"}},
            )

        assert result["status"] == "rejected"
        assert result["human_decision"] == "reject"
