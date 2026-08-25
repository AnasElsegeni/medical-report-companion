import json
from pathlib import Path

from backend.report.extractor import extract_text_from_pdf
from backend.report.parser import parse_report


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "medical_report.pdf"


def main():

    print("=" * 60)
    print("MEDICAL REPORT PARSER TEST")
    print("=" * 60)

    try:

        # Step 1: Extract text from PDF
        text = extract_text_from_pdf(PDF_PATH)

        print("\nText extraction successful.")

        # Step 2: Parse report
        report = parse_report(text)

        print("\nReport parsing successful.")

        # Step 3: Display structured report
        print("\n" + "=" * 60)
        print("STRUCTURED REPORT")
        print("=" * 60)

        print(
            json.dumps(
                report,
                indent=4,
                ensure_ascii=False
            )
        )

        print("\n" + "=" * 60)
        print(f"Number of tests detected: {len(report['tests'])}")
        print("=" * 60)

    except Exception as error:

        print("\nERROR:")
        print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()