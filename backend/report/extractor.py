from pathlib import Path

import pymupdf


def extract_text_from_pdf(file_path: str | Path) -> str:
    """
    Extract text from a text-based PDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    text_parts = []

    with pymupdf.open(file_path) as document:

        for page_number, page in enumerate(
            document,
            start=1
        ):

            page_text = page.get_text("text").strip()

            if page_text:
                text_parts.append(
                    f"--- Page {page_number} ---\n"
                    f"{page_text}"
                )

    return "\n\n".join(text_parts).strip()
  