from pathlib import Path

from backend.report.extractor import extract_text_from_pdf


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# PDF location
PDF_PATH = BASE_DIR / "data" / "medical_report.pdf"


def main():

    print("=" * 60)
    print("MEDICAL REPORT TEXT EXTRACTION TEST")
    print("=" * 60)

    print(f"\nPDF: {PDF_PATH}")

    try:

        text = extract_text_from_pdf(PDF_PATH)

        if not text:
            print("\nNo text was extracted.")
            return

        print("\nExtraction successful.")

        print("\n" + "=" * 60)
        print("EXTRACTED TEXT")
        print("=" * 60)

        print(text)

        print("\n" + "=" * 60)
        print(f"Characters extracted: {len(text)}")
        print("=" * 60)

    except Exception as error:

        print("\nERROR:")
        print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()