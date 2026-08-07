from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.services.nodes import (
    generate_node,
    repair_node,
    retrieve_node,
    validate_node,
)
from app.state import AgentState


def route_after_validation(
    state: AgentState,
) -> Literal["end", "repair"]:
    if state["is_valid"]:
        return "end"

    settings = get_settings()

    if state["repair_attempts"] >= settings.max_repair_attempts:
        return "end"

    return "repair"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("repair", repair_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")

    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "end": END,
            "repair": "repair",
        },
    )

    workflow.add_edge("repair", "validate")

    return workflow.compile()


agent_graph = build_graph()
