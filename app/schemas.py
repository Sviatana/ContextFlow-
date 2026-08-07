from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Question to answer using retrieved context.",
    )


class GeneratedAnswer(BaseModel):
    answer: str = Field(
        ...,
        min_length=1,
        description="Answer grounded only in the supplied context.",
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    context: list[str]
    is_valid: bool
    errors: list[str]
    repair_attempts: int
