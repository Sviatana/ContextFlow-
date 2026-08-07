# ContextFlow

**RAG backend with FastAPI, LangChain and LangGraph**

ContextFlow is a backend service demonstrating a controlled LLM workflow built around retrieval, structured generation, validation and conditional repair.

The project uses LangGraph `StateGraph` for orchestration and LangChain for prompts, embeddings, model calls and structured LLM output.

## Workflow

```text
START
  |
  v
retrieve
  |
  v
generate
  |
  v
validate
  | \
  |  \ invalid
  |   v
  | repair
  |   |
  |   +------> validate
  |
  +----------> END
       valid
```

The workflow state contains:

- question
- retrieved context
- generated answer
- validation status
- validation errors
- repair attempt count

## Architecture

```text
FastAPI
   |
   v
LangGraph StateGraph
   |
   +--> retrieve
   |      |
   |      +--> OpenAIEmbeddings
   |      +--> cosine similarity
   |
   +--> generate
   |      |
   |      +--> ChatPromptTemplate
   |      +--> ChatOpenAI
   |      +--> Pydantic structured output
   |
   +--> validate
          |
          +--> valid -> END
          |
          +--> invalid -> repair
                            |
                            +--> validate
```

## Engineering Highlights

- LangGraph `StateGraph`
- Explicit graph nodes and edges
- Conditional routing
- Bounded validation/repair loop
- Typed workflow state
- LangChain prompt composition
- ChatOpenAI integration
- OpenAI embeddings
- Pydantic structured output
- Semantic retrieval
- Cosine similarity
- Explicit response validation
- FastAPI API layer
- Environment-based configuration
- Unit tests for graph routing and validation

## Graph

The workflow is defined in `app/graph.py`.

The main nodes are `retrieve`, `generate`, `validate`, and `repair`.

Normal path:

```text
retrieval -> generation -> validation -> END
```

Repair path:

```text
retrieval -> generation -> validation -> repair -> validation -> END
```

Repair attempts are bounded by configuration to prevent an uncontrolled loop.

## Project Structure

```text
app/
├── config.py
├── graph.py
├── main.py
├── schemas.py
├── state.py
├── data/
│   └── documents.py
└── services/
    ├── llm.py
    ├── nodes.py
    ├── retrieval.py
    └── validation.py

tests/
├── test_graph.py
└── test_validation.py
```

## API

- `GET /health`
- `POST /ask`

Example request:

```json
{
  "question": "What is LangGraph used for?"
}
```

The backend initializes graph state, retrieves semantically relevant context, generates a structured answer, validates it, conditionally routes invalid output through repair, validates again, and returns the final state.

## Local Setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Run tests and linting:

```bash
pytest -q
ruff check app tests
```

## Tech Stack

- Python 3.12
- FastAPI
- Pydantic 2
- LangGraph
- LangChain
- LangChain OpenAI
- OpenAI API
- Embeddings
- Semantic retrieval
- Cosine similarity

## Scope

The repository intentionally keeps the knowledge base small and in memory so the orchestration and retrieval logic remain easy to inspect.

The retrieval layer can later be replaced by PostgreSQL/pgvector or another vector store without changing the graph-level workflow.
