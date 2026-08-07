from app.services.llm import generate_answer, repair_answer
from app.services.retrieval import retrieve_context
from app.services.validation import validate_answer
from app.state import AgentState


def retrieve_node(state: AgentState) -> dict:
    context = retrieve_context(state["question"])

    return {
        "context": context,
        "errors": [],
    }


def generate_node(state: AgentState) -> dict:
    answer = generate_answer(
        state["question"],
        state["context"],
    )

    return {
        "answer": answer,
    }


def validate_node(state: AgentState) -> dict:
    errors = validate_answer(
        question=state["question"],
        answer=state["answer"],
        context=state["context"],
    )

    return {
        "is_valid": not errors,
        "errors": errors,
    }


def repair_node(state: AgentState) -> dict:
    repaired_answer = repair_answer(
        question=state["question"],
        context=state["context"],
        answer=state["answer"],
        errors=state["errors"],
    )

    return {
        "answer": repaired_answer,
        "repair_attempts": state["repair_attempts"] + 1,
        "is_valid": False,
        "errors": [],
    }
