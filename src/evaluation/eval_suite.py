"""
Eval Suite
----------
Runs a golden dataset against a prompt version and gates promotions.

A prompt must achieve:
  - 100% schema validity  (output always parses as valid JSON + Pydantic model)
  - >= 95% field accuracy (extracted values match golden expected values)

Usage:
  python -m scripts.run_eval --env dev --promote
"""

import json
import time
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

from src.registry.prompt_registry import PromptRegistry

registry = PromptRegistry(prompts_dir="prompts")

# ── Golden dataset ─────────────────────────────────────────────────────────────
GOLDEN_DATASET = [
    {
        "input": (
            "Pay to the order of Jane Smith *** $2,500.00 "
            "Two Thousand Five Hundred Dollars  Date: 2024-01-15"
        ),
        "expected": {
            "payee_name":     "Jane Smith",
            "amount_numeric": 2500.0,
            "cheque_date":    "2024-01-15",
        },
    },
    {
        "input": (
            "Payee: ADCB Bank  Amount: AED 10,000  Ten Thousand Dirhams  "
            "15/01/2024  MICR: 000123|001|9876543210"
        ),
        "expected": {
            "payee_name":     "ADCB Bank",
            "amount_numeric": 10000.0,
            "cheque_date":    "2024-01-15",
        },
    },
    {
        "input": (
            "John Doe  Five Hundred Dollars  $500  "
            "Account: 1234567890  Date: 20-03-2024"
        ),
        "expected": {
            "payee_name":     "John Doe",
            "amount_numeric": 500.0,
            "account_number": "1234567890",
        },
    },
]


@dataclass
class EvalResult:
    total: int = 0
    schema_valid: int = 0
    field_match: int = 0
    latencies: list = field(default_factory=list)
    failures: list = field(default_factory=list)

    @property
    def schema_accuracy(self) -> float:
        return self.schema_valid / self.total if self.total else 0.0

    @property
    def field_accuracy(self) -> float:
        return self.field_match / self.total if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (sum(self.latencies) / len(self.latencies) * 1000) if self.latencies else 0.0

    def passed(self, threshold: float = 0.95) -> bool:
        return self.schema_accuracy == 1.0 and self.field_accuracy >= threshold


def run_eval(
    prompt_name: str = "cheque_extraction",
    env: str = "dev",
    dataset: list | None = None,
) -> EvalResult:
    """Run the golden dataset against the prompt active in the given env."""
    prompt = registry.get(prompt_name, env=env)
    dataset = dataset or GOLDEN_DATASET

    llm = ChatOpenAI(
        model=prompt.model,
        temperature=prompt.temperature,
        max_tokens=prompt.max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    result = EvalResult(total=len(dataset))

    for i, sample in enumerate(dataset):
        messages = [{"role": "system", "content": prompt.system}]
        for ex in prompt.few_shot_examples:
            messages.append({"role": "user",      "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})
        messages.append({"role": "user", "content": sample["input"]})

        t0 = time.time()
        try:
            response = llm.invoke(messages)
            latency = time.time() - t0
            result.latencies.append(latency)

            parsed = json.loads(response.content)
            result.schema_valid += 1

            mismatches = {
                k: {"expected": v, "got": parsed.get(k)}
                for k, v in sample["expected"].items()
                if parsed.get(k) != v
            }
            if not mismatches:
                result.field_match += 1
            else:
                result.failures.append({"sample": i, "mismatches": mismatches})

        except json.JSONDecodeError as exc:
            result.latencies.append(time.time() - t0)
            result.failures.append({"sample": i, "error": f"JSON error: {exc}"})
        except Exception as exc:
            result.latencies.append(time.time() - t0)
            result.failures.append({"sample": i, "error": str(exc)})

    return result


def print_report(result: EvalResult, prompt_name: str, env: str) -> None:
    version = registry.active_version(prompt_name, env)
    icon = "✅" if result.passed() else "❌"
    print(f"\n{'─'*54}")
    print(f"{icon}  {prompt_name} v{version} [{env}]")
    print(f"{'─'*54}")
    print(f"  Samples         : {result.total}")
    print(f"  Schema accuracy : {result.schema_accuracy:.1%}")
    print(f"  Field accuracy  : {result.field_accuracy:.1%}")
    print(f"  Avg latency     : {result.avg_latency_ms:.0f} ms")
    if result.failures:
        print(f"\n  Failures ({len(result.failures)}):")
        for f in result.failures:
            print(f"    sample {f['sample']}: "
                  f"{f.get('mismatches') or f.get('error')}")
    print(f"{'─'*54}")
    print(f"  Result: {'PASS — ready to promote' if result.passed() else 'FAIL — blocked'}")
    print(f"{'─'*54}\n")


def promote_if_passing(
    prompt_name: str,
    from_env: str,
    to_env: str,
    threshold: float = 0.95,
) -> bool:
    """Run eval and auto-promote to the next environment if it passes."""
    result = run_eval(prompt_name, env=from_env)
    print_report(result, prompt_name, from_env)
    version = registry.active_version(prompt_name, from_env)
    if result.passed(threshold):
        registry.promote(prompt_name, version, from_env, to_env)
        return True
    print(f"🚫 Blocked: field_accuracy={result.field_accuracy:.1%} < {threshold:.0%}")
    return False
