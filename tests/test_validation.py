from app.services.validation import validate_answer


def test_empty_answer_is_rejected():
    errors = validate_answer(
        question="What is RAG?",
        answer="",
        context=["RAG combines retrieval and generation."],
    )

    assert "answer is empty" in errors


def test_grounded_answer_is_valid():
    errors = validate_answer(
        question="What is RAG?",
        answer=(
            "Retrieval-augmented generation combines retrieval "
            "with language-model generation using supplied context."
        ),
        context=[
            "Retrieval-augmented generation combines information retrieval "
            "with language-model generation."
        ],
    )

    assert errors == []


def test_unrelated_answer_is_rejected():
    errors = validate_answer(
        question="What is RAG?",
        answer=(
            "A database transaction groups several database operations "
            "into one atomic unit."
        ),
        context=[
            "Retrieval-augmented generation combines information retrieval "
            "with language-model generation."
        ],
    )

    assert "answer has no meaningful lexical overlap with retrieved context" in errors
