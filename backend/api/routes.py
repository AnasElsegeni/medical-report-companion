from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.generation.generator import generate_answer


router = APIRouter(
    prefix="/api",
    tags=["Medical Q&A"]
)


class QuestionRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    result = generate_answer(question)

    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"]
    }
    