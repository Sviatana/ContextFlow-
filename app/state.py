from typing import TypedDict


class AgentState(TypedDict):
    question: str
    context: list[str]
    answer: str
    is_valid: bool
    errors: list[str]
    repair_attempts: int
