import re


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_-]+", value.lower())
        if len(token) >= 4
    }


def validate_answer(
    *,
    question: str,
    answer: str,
    context: list[str],
) -> list[str]:
    """Validate generated output before it leaves the RAG workflow.

    Semantic relevance between the question and retrieved documents belongs
    to the retrieval layer. This validator checks application-level output
    quality and whether the generated answer is grounded in retrieved context.
    """
    errors: list[str] = []

    if not question.strip():
        errors.append("question is empty")

    normalized = answer.strip()

    if not normalized:
        errors.append("answer is empty")
        return errors

    if len(normalized) < 25:
        errors.append("answer is too short")

    if not context:
        errors.append("retrieval returned no context")
        return errors

    context_tokens = _tokens(" ".join(context))
    answer_tokens = _tokens(normalized)

    if context_tokens and answer_tokens:
        overlap = context_tokens & answer_tokens

        if not overlap:
            errors.append(
                "answer has no meaningful lexical overlap with retrieved context"
            )

    return errors
