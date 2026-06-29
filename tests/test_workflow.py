"""
Unit tests for the Banking AI cheque-processing pipeline.

Run with:  pytest tests/ -v

These tests use mocked LLM responses so no OpenAI key is needed.
"""

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
        # dev is on 1.2.0 — promote it to staging
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
            confidence=0.5,   # below 0.80 threshold
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


# ── Workflow integration test (mocked LLM) ────────────────────────────────────

class TestWorkflow:

    GOOD_LLM_RESPONSE = (
        '{"amount_numeric":1500.0,"amount_words":"One Thousand Five Hundred",'
        '"payee_name":"Jane Smith","cheque_date":"2024-01-15",'
        '"account_number":null,"bank_name":null,"micr_line":null,'
        '"confidence_note":"Clear extraction"}'
    )

    @patch("src.agents.nodes.ChatOpenAI")
    def test_auto_approve_path(self, MockLLM):
        """High-confidence cheque should auto-approve without HITL."""
        mock_response = MagicMock()
        mock_response.content = self.GOOD_LLM_RESPONSE
        MockLLM.return_value.invoke.return_value = mock_response

        from src.agents.workflow import build_workflow
        app = build_workflow()

        config = {"configurable": {"thread_id": "test-auto-001"}}
        result = app.invoke(
            {
                "cheque_id":   "CHQ-001",
                "cheque_text": "Pay to Jane Smith $1500 One Thousand Five Hundred Date: 2024-01-15",
                "audit_log":   [],
            },
            config=config,
        )

        assert result["status"] == "approved"
        assert result["human_decision"] == "approve"
        assert result["final_fields"]["payee_name"] == "Jane Smith"
        assert len(result["audit_log"]) >= 4   # extract, validate, auto_approve, finalise
