# Banking AI Assistant 🏦

A production-grade **Intelligent Document Processing (IDP)** pipeline for banking cheque clearing, built with LangGraph, LangChain, and Azure AI services.

This project demonstrates the core engineering patterns required in an AI Engineering role at a Banking AI Centre of Excellence — agentic workflows, Human-in-the-Loop (HITL), prompt versioning & governance, schema validation, and model evaluation frameworks.

---

## Architecture

```
cheque image / text
       │
       ▼
┌─────────────┐     versioned YAML     ┌──────────────────┐
│   Extract   │◄──────────────────────│  Prompt Registry  │
│   (LLM)     │                        │  v1.0 / v1.1 / v1.2│
└──────┬──────┘                        └──────────────────┘
       │ extracted_fields + confidence
       ▼
┌─────────────┐
│  Validate   │  business rules: missing fields, amount mismatch
└──────┬──────┘
       │
  ┌────┴─────────────┐
  │ confidence < 0.8 │
  │  OR errors?      │
  └────┬─────────────┘
       │YES                    NO
       ▼                       ▼
┌─────────────┐        ┌──────────────┐
│ HITL Review │        │ Auto-Approve │
│ interrupt() │        │              │
└──────┬──────┘        └──────┬───────┘
       │ Command(resume=...)   │
       └──────────┬────────────┘
                  ▼
           ┌────────────┐
           │  Finalise  │  write to core banking
           └────────────┘
```

---

## Key Concepts Demonstrated

### 1. LangGraph Agentic Workflow
- `StateGraph` with typed state (`ChequeState`)
- Fixed and conditional edges for routing
- `MemorySaver` checkpointer enabling graph pause/resume

### 2. Human-in-the-Loop (HITL)
- `interrupt()` pauses the graph when confidence < 0.80 or validation errors exist
- Graph state is serialised to a checkpointer
- Reviewer decision is injected via `Command(resume={...})`
- Graph resumes from the exact point it paused using `thread_id`

### 3. Prompt Versioning & Governance
- Every prompt lives in a versioned YAML file (`prompts/cheque_extraction/v1.x.x.yaml`)
- `PromptRegistry` maps each environment (dev / staging / production) to a pinned version
- Promotion (`dev → staging → production`) only happens after the eval suite passes
- Rollback is a one-line call: `registry.rollback(...)`

### 4. Schema Validation
- LLM output is enforced as JSON (`response_format: json_object`)
- Pydantic model (`ChequeFields`) validates types, nullability, and business constraints
- Invalid output is caught and routed to error handling — never silently passed through

### 5. Model Evaluation Framework
- Golden dataset of labelled cheque samples
- Eval reports: schema accuracy, field accuracy, avg latency
- A prompt is blocked from promotion if field accuracy < 95%

### 6. Mock Azure Document Intelligence
- `src/tools/mock_azure.py` simulates the `prebuilt-check` model
- Drop-in replacement for the real `azure-ai-documentintelligence` SDK
- Allows the full pipeline to run locally without Azure credentials

---

## Project Structure

```
banking-ai-assistant/
├── prompts/
│   └── cheque_extraction/
│       ├── v1.0.0.yaml          # production — stable
│       ├── v1.1.0.yaml          # staging — few-shot examples added
│       └── v1.2.0.yaml          # dev — MICR + date normalisation rules
│
├── src/
│   ├── agents/
│   │   ├── state.py             # ChequeState TypedDict
│   │   ├── nodes.py             # extract, validate, hitl_review, finalise
│   │   └── workflow.py          # StateGraph builder
│   ├── registry/
│   │   └── prompt_registry.py  # versioned prompt loader + promotion logic
│   ├── evaluation/
│   │   └── eval_suite.py        # golden dataset + eval runner + auto-promote
│   └── tools/
│       └── mock_azure.py        # Azure Document Intelligence simulator
│
├── tests/
│   └── test_workflow.py         # unit tests (no LLM key needed — mocked)
│
├── scripts/
│   ├── run_cheque.py            # demo: auto-approve and HITL paths
│   └── run_eval.py              # run eval suite, optionally promote
│
├── .env.example                 # environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Aravindalochanan1996/project-using-langchain-langgraph.git
cd project-using-langchain-langgraph
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the demo

**Auto-approve path** (high-confidence cheque, no human needed):
```bash
python scripts/run_cheque.py
```

**HITL path** (low-confidence cheque, reviewer intervenes):
```bash
python scripts/run_cheque.py --hitl
```

**Use a specific environment's prompt version:**
```bash
APP_ENV=staging python scripts/run_cheque.py
```

### 4. Run tests (no API key needed — LLM is mocked)

```bash
pytest tests/ -v
```

> **Corporate laptop / Application Control policy error?**
> If you see `ImportError: DLL load failed while importing _uuid_utils: An Application Control policy has blocked this file`, that's your machine's security policy (e.g. Windows Defender Application Control) blocking a native DLL inside `uuid_utils`, a transitive dependency pulled in by `langsmith`'s pytest plugin — unrelated to this project's actual code. It's already disabled via `pytest.ini` (`-p no:langsmith`). If it still occurs, run `pip uninstall langsmith uuid_utils -y`; neither package is required for this project to function.

### 5. Run the eval suite

```bash
# Evaluate dev prompt version
python scripts/run_eval.py --env dev

# Evaluate and auto-promote to staging if passing
python scripts/run_eval.py --env dev --promote --to staging
```

---

## Prompt Governance Lifecycle

```
Author YAML  →  Git PR  →  run_eval (dev)  →  promote to staging
                                 │
                          field_accuracy < 95%?
                                 │
                            BLOCKED ✗
                                 │
                          field_accuracy >= 95%?
                                 │
                         promote to staging ✓
                                 │
                          shadow traffic test
                                 │
                         promote to production ✓
                                 │
                         accuracy drops in prod?
                                 │
                         rollback instantly ⏪
```

---

## Environment → Prompt Version Mapping

| Environment | Active Version | Notes |
|-------------|---------------|-------|
| production  | v1.0.0        | Zero-shot, battle-tested |
| staging     | v1.1.0        | + few-shot examples |
| dev         | v1.2.0        | + MICR rules, date normalisation |

---

## Extending to Real Azure Services

To replace the mock Azure module with the real SDK:

```python
# src/tools/azure_doc_intel.py
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import os

client = DocumentIntelligenceClient(
    endpoint=os.getenv("AZURE_DOC_INTEL_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_DOC_INTEL_KEY")),
)

def analyze_cheque(image_bytes: bytes):
    poller = client.begin_analyze_document(
        "prebuilt-check",
        analyze_request=image_bytes,
        content_type="image/jpeg",
    )
    return poller.result()
```

Then import `analyze_cheque` from this module instead of `mock_azure` in `nodes.py`.

---

## Technologies

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph |
| LLM framework | LangChain |
| LLM | OpenAI GPT-4o / Azure OpenAI |
| Schema validation | Pydantic v2 |
| Document extraction | Azure Document Intelligence (mocked) |
| Prompt storage | YAML + Git |
| Testing | pytest |
| Language | Python 3.12 |

---

## Interview Talking Points

- **Why LangGraph over plain LangChain?** LangGraph gives you explicit state, conditional routing, and a checkpointer — essential for HITL and auditability in banking.
- **Why YAML for prompts?** Prompts are config, not code. YAML gives you Git history, PR review, and environment pinning without redeployment.
- **How do you debug non-determinism?** Freeze temperature at 0.1, run the same input 10 times against the golden dataset, and track variance in field accuracy across prompt versions.
- **What would production look like?** Replace `MemorySaver` with `SqliteSaver` or a Postgres checkpointer, deploy the graph as an Azure Container App, connect the HITL reviewer UI via webhooks, and wire the finalise node to the core banking API.
