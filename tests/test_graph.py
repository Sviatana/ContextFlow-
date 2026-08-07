from app.config import get_settings
from app.graph import agent_graph, route_after_validation


def base_state():
    return {
        "question": "What is LangGraph?",
        "context": ["LangGraph builds stateful graph workflows."],
        "answer": "LangGraph builds stateful graph workflows for LLM applications.",
        "is_valid": False,
        "errors": ["test error"],
        "repair_attempts": 0,
    }


def test_graph_compiles_with_expected_nodes():
    graph = agent_graph.get_graph()

    assert "retrieve" in graph.nodes
    assert "generate" in graph.nodes
    assert "validate" in graph.nodes
    assert "repair" in graph.nodes


def test_valid_answer_routes_to_end():
    state = base_state()
    state["is_valid"] = True

    assert route_after_validation(state) == "end"


def test_invalid_answer_routes_to_repair():
    get_settings.cache_clear()

    state = base_state()

    assert route_after_validation(state) == "repair"


def test_repair_limit_routes_to_end():
    settings = get_settings()

    state = base_state()
    state["repair_attempts"] = settings.max_repair_attempts

    assert route_after_validation(state) == "end"
