from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from backend.generation.generator import generate_answer
from backend.report.extractor import extract_text_from_pdf
from backend.report.parser import parse_report

router = APIRouter(
    prefix="/api",
    tags=["Medical Q&A"]
)


class QuestionRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(request: QuestionRequest):

    from backend.generation.generator import generate_answer

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
    
@router.post("/report/upload")
async def upload_report(file: UploadFile = File(...)):

    # Make sure the uploaded file is a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:
        # Read uploaded file
        file_content = await file.read()

        # Save temporarily
        import tempfile

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_content)
            temp_file_path = temp_file.name

        # Extract text
        text = extract_text_from_pdf(temp_file_path)

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the PDF."
            )

        # Parse medical report
        report = parse_report(text)

        return {
            "filename": file.filename,
            "report": report
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process report: {str(error)}"
        )
    