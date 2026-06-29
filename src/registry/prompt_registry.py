"""
Prompt Registry
---------------
Single source of truth for versioned prompts.

All LLM calls load their system prompt through this registry.
Hardcoded prompt strings in application code are an anti-pattern —
they can't be versioned, tested, or rolled back independently.

Environment pinning
-------------------
Each prompt has an active version per environment:
  dev        → latest / in-progress
  staging    → passed eval, under shadow-traffic testing
  production → stable, battle-tested

In production, _active_versions would live in a config service
(e.g. Azure App Configuration) so promotions don't need redeployments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Prompt:
    version: str
    name: str
    system: str
    model: str
    temperature: float
    max_tokens: int
    few_shot_examples: list = field(default_factory=list)


class PromptRegistry:

    # Maps environment → prompt_name → active version string
    _active_versions: dict = {
        "dev":        {"cheque_extraction": "1.2.0"},
        "staging":    {"cheque_extraction": "1.1.0"},
        "production": {"cheque_extraction": "1.0.0"},
    }

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)

    def get(self, name: str, env: str | None = None) -> Prompt:
        """Return the Prompt object active in the given environment."""
        env = env or os.getenv("APP_ENV", "production")
        version = self._active_versions.get(env, {}).get(name)
        if not version:
            raise ValueError(f"No active version for prompt='{name}' env='{env}'")

        path = self.prompts_dir / name / f"v{version}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return Prompt(
            version=data["version"],
            name=data["name"],
            system=data["system"].strip(),
            model=data["model"]["name"],
            temperature=data["model"]["temperature"],
            max_tokens=data["model"]["max_tokens"],
            few_shot_examples=data.get("few_shot_examples", []),
        )

    def active_version(self, name: str, env: str) -> str:
        return self._active_versions.get(env, {}).get(name, "unknown")

    def promote(self, name: str, version: str, from_env: str, to_env: str) -> None:
        """Move a version to the next environment after it passes eval."""
        current = self._active_versions.get(from_env, {}).get(name)
        if current != version:
            raise ValueError(
                f"v{version} is not active in {from_env} (active: {current})"
            )
        self._active_versions.setdefault(to_env, {})[name] = version
        print(f"✅ Promoted '{name}' v{version}: {from_env} → {to_env}")

    def rollback(self, name: str, env: str, to_version: str) -> None:
        """Instantly revert an environment to a known-good version."""
        self._active_versions.setdefault(env, {})[name] = to_version
        print(f"⏪ Rolled back '{name}' in {env} to v{to_version}")
