from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.schemas import GeneratedAnswer

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grounded question-answering component inside a RAG pipeline. "
            "Answer only from the supplied context. "
            "If the context is insufficient, say that the available context is insufficient. "
            "Do not invent facts.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Return a concise grounded answer.",
        ),
    ]
)


REPAIR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You repair an answer that failed application validation. "
            "Use only the supplied context and correct the validation problems. "
            "Do not invent information.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "Previous answer:\n{answer}\n\n"
            "Validation errors:\n{errors}\n\n"
            "Return a corrected answer.",
        ),
    ]
)


@lru_cache
def get_chat_model() -> ChatOpenAI:
    settings = get_settings()

    kwargs: dict[str, object] = {
        "model": settings.chat_model,
        "temperature": 0,
        "api_key": settings.openai_api_key,
    }

    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    return ChatOpenAI(**kwargs)


def generate_answer(question: str, context: list[str]) -> str:
    model = get_chat_model().with_structured_output(GeneratedAnswer)

    chain = GENERATION_PROMPT | model

    result = chain.invoke(
        {
            "question": question,
            "context": "\n\n".join(context),
        }
    )

    return result.answer


def repair_answer(
    *,
    question: str,
    context: list[str],
    answer: str,
    errors: list[str],
) -> str:
    model = get_chat_model().with_structured_output(GeneratedAnswer)

    chain = REPAIR_PROMPT | model

    result = chain.invoke(
        {
            "question": question,
            "context": "\n\n".join(context),
            "answer": answer,
            "errors": "\n".join(f"- {error}" for error in errors),
        }
    )

    return result.answer
