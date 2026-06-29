"""
Cheque Processing Workflow
--------------------------
Builds and compiles the LangGraph StateGraph.

Flow:
  extract → validate → [route] → hitl_review  ──┐
                               → auto_approve ──┤
                                                 ↓
                                             finalise → END
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.nodes import (
    auto_approve_node,
    extract_fields_node,
    finalise_node,
    hitl_review_node,
    validate_fields_node,
)
from src.agents.state import ChequeState


def _route_after_validation(state: ChequeState) -> str:
    """
    Conditional edge: send to HITL if confidence is low or errors exist,
    otherwise auto-approve.
    """
    return "hitl_review" if state.get("needs_human_review") else "auto_approve"


def build_workflow(checkpointer=None):
    """
    Compile the LangGraph cheque-processing workflow.

    Args:
        checkpointer: A LangGraph checkpointer instance.
                      MemorySaver is used by default (in-process, no persistence).
                      Use SqliteSaver or a Postgres checkpointer in production
                      so HITL pauses survive server restarts.

    Returns:
        Compiled LangGraph CompiledGraph ready to invoke.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    g = StateGraph(ChequeState)

    g.add_node("extract",      extract_fields_node)
    g.add_node("validate",     validate_fields_node)
    g.add_node("hitl_review",  hitl_review_node)
    g.add_node("auto_approve", auto_approve_node)
    g.add_node("finalise",     finalise_node)

    g.set_entry_point("extract")

    g.add_edge("extract",      "validate")
    g.add_edge("hitl_review",  "finalise")
    g.add_edge("auto_approve", "finalise")
    g.add_edge("finalise",     END)

    g.add_conditional_edges(
        "validate",
        _route_after_validation,
        {"hitl_review": "hitl_review", "auto_approve": "auto_approve"},
    )

    return g.compile(checkpointer=checkpointer)
