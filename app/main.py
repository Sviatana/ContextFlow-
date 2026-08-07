from fastapi import FastAPI, HTTPException

from app.graph import agent_graph
from app.schemas import AskRequest, AskResponse

app = FastAPI(
    title="ContextFlow",
    description=(
        "RAG backend built with FastAPI, LangChain and LangGraph. "
        "The workflow performs retrieval, structured generation, validation "
        "and conditional repair."
    ),
    version="2.0.0",
)


def initial_state(question: str) -> dict:
    return {
        "question": question,
        "context": [],
        "answer": "",
        "is_valid": False,
        "errors": [],
        "repair_attempts": 0,
    }


def run_agent(question: str) -> dict:
    return agent_graph.invoke(initial_state(question))


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "service": "contextflow",
        "workflow": "langgraph",
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        result = run_agent(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="AI workflow execution failed",
        ) from exc

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        context=result["context"],
        is_valid=result["is_valid"],
        errors=result["errors"],
        repair_attempts=result["repair_attempts"],
    )
