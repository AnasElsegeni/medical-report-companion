import re
from typing import Any


def parse_patient_info(text: str) -> dict[str, Any]:
    """
    Extract basic patient information from the report text.
    """

    patient = {
        "name": None,
        "age": None,
        "sex": None,
        "report_date": None,
    }

    # Patient Name
    name_match = re.search(
        r"Patient Name:\s*(.+)",
        text,
        re.IGNORECASE
    )

    if name_match:
        patient["name"] = name_match.group(1).strip()

    # Age
    age_match = re.search(
        r"Age:\s*(\d+)",
        text,
        re.IGNORECASE
    )

    if age_match:
        patient["age"] = int(age_match.group(1))

    # Sex
    sex_match = re.search(
        r"Sex:\s*(.+)",
        text,
        re.IGNORECASE
    )

    if sex_match:
        patient["sex"] = sex_match.group(1).strip()

    # Report Date
    date_match = re.search(
        r"Report Date:\s*(.+)",
        text,
        re.IGNORECASE
    )

    if date_match:
        patient["report_date"] = date_match.group(1).strip()

    return patient


def parse_reference_range(reference: str) -> tuple[float | None, float | None]:
    """
    Convert a reference range into lower and upper values.

    Examples:
        "0.7 – 1.3" -> (0.7, 1.3)
        "4.0 - 11.0" -> (4.0, 11.0)
        ">60" -> (60, None)
        "<200" -> (None, 200)
    """

    reference = reference.strip()

    # Range: 0.7 - 1.3
    range_match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)",
        reference
    )

    if range_match:
        return (
            float(range_match.group(1)),
            float(range_match.group(2))
        )

    # Greater than: >60
    greater_match = re.search(
        r">\s*([0-9]+(?:\.[0-9]+)?)",
        reference
    )

    if greater_match:
        return (
            float(greater_match.group(1)),
            None
        )

    # Less than: <200
    less_match = re.search(
        r"<\s*([0-9]+(?:\.[0-9]+)?)",
        reference
    )

    if less_match:
        return (
            None,
            float(less_match.group(1))
        )

    return None, None


def determine_status(
    value: float,
    reference_low: float | None,
    reference_high: float | None
) -> str:
    """
    Determine whether a laboratory result is low, normal, or high.
    """

    if reference_low is not None and value < reference_low:
        return "low"

    if reference_high is not None and value > reference_high:
        return "high"

    return "normal"

def normalize_report_text(text: str) -> str:
    """
    Normalize common formatting issues in extracted PDF text.
    """

    text = text.replace(
        "Blood Urea Nitrogen \n(BUN)",
        "Blood Urea Nitrogen (BUN)"
    )

    return text
  
def parse_lab_results(text: str) -> list[dict[str, Any]]:
    """
    Extract laboratory test results from the report text.

    This first version is designed for our current test report.
    """

    tests = []

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Known test names from our current report.
    test_names = [
        "White Blood Cells (WBC)",
        "Red Blood Cells (RBC)",
        "Hemoglobin (Hb)",
        "Hematocrit (HCT)",
        "Platelets",
        "Creatinine",
        "Blood Urea Nitrogen (BUN)",
        "eGFR",
        "HbA1c",
        "Fasting Blood Glucose",
        "TSH",
        "Total Cholesterol",
        "LDL Cholesterol",
        "HDL Cholesterol",
        "Triglycerides",
        "ALT",
        "AST",
        "ALP",
        "Total Bilirubin",
    ]

    for index, line in enumerate(lines):

        if line not in test_names:
            continue

        test_name = line

        # We expect:
        # test name
        # result
        # reference range
        # unit

        if index + 3 >= len(lines):
            continue

        result_text = lines[index + 1]
        reference_text = lines[index + 2]
        unit = lines[index + 3]

        # Convert result to float
        try:
            value = float(result_text)
        except ValueError:
            continue

        reference_low, reference_high = parse_reference_range(
            reference_text
        )

        status = determine_status(
            value,
            reference_low,
            reference_high
        )

        tests.append({
            "name": test_name,
            "value": value,
            "unit": unit,
            "reference_low": reference_low,
            "reference_high": reference_high,
            "status": status,
        })

    return tests


def parse_report(text: str) -> dict[str, Any]:
    """
    Parse the complete medical report.
    """

    text = normalize_report_text(text)

    return {
        "patient": parse_patient_info(text),
        "tests": parse_lab_results(text),
    }