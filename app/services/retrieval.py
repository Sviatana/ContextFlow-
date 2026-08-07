from functools import lru_cache

import numpy as np
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.data.documents import DOCUMENTS


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0

    return float(np.dot(a, b) / denominator)


@lru_cache
def get_embeddings_client() -> OpenAIEmbeddings:
    settings = get_settings()

    kwargs: dict[str, object] = {
        "model": settings.embedding_model,
        "api_key": settings.openai_api_key,
    }

    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    return OpenAIEmbeddings(**kwargs)


def retrieve_context(question: str) -> list[str]:
    settings = get_settings()
    embeddings = get_embeddings_client()

    document_texts = [text for _, text in DOCUMENTS]

    query_vector = embeddings.embed_query(question)
    document_vectors = embeddings.embed_documents(document_texts)

    ranked = sorted(
        zip(document_texts, document_vectors, strict=True),
        key=lambda item: cosine_similarity(query_vector, item[1]),
        reverse=True,
    )

    return [
        document
        for document, _ in ranked[: settings.retrieval_top_k]
    ]
