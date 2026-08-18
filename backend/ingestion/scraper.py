import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_FILE = BASE_DIR / "data" / "sources.json"
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"


def load_sources():
    with open(SOURCES_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["sources"]


def download_page(url: str) -> str:
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0 MedicalReportCompanion/0.1"
        },
    )

    response.raise_for_status()

    response.encoding = "utf-8"

    return response.text


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that are not useful for the knowledge base.
    for element in soup([
        "script",
        "style",
        "noscript",
        "nav",
        "footer",
        "header",
        "form",
        "aside",
    ]):
        element.decompose()

    # Prefer article content, then main, then body.
    content = (
        soup.find("article")
        or soup.find("main")
        or soup.body
    )

    if content is None:
        return ""

    # Remove navigation and utility elements.
    for element in content.find_all(
        class_=lambda value: value and any(
            keyword in str(value).lower()
            for keyword in [
                "navigation",
                "breadcrumb",
                "menu",
                "header",
                "footer",
                "utility",
                "share",
            ]
        )
    ):
        element.decompose()

    blocks = []

    # Extract meaningful blocks while preserving document structure.
    for element in content.find_all([
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
    ]):

        text = element.get_text(" ", strip=True)

        if not text:
            continue

        # Remove common website boilerplate.
        if text in {
            "Skip navigation",
            "Official websites use .gov",
            "A",
            ".gov",
            "You Are Here:",
            "Home",
            "Medical Tests",
            "Share",
        }:
            continue

        blocks.append(text)

    # Remove consecutive duplicate blocks.
    cleaned_blocks = []

    for block in blocks:
        if not cleaned_blocks or block != cleaned_blocks[-1]:
            cleaned_blocks.append(block)

    text = "\n\n".join(cleaned_blocks)
        # Normalize whitespace artifacts produced by HTML extraction.
    text = text.replace("agroup", "a group")
    text = text.replace("oxygenfrom", "oxygen from")
    text = text.replace("partof", "part of")
    text = text.replace("bloodcell", "blood cell")
    text = text.replace("completeblood", "complete blood")
    text = text.replace("collectedinto", "collected into")
    text = text.replace("takea", "take a")
    text = text.replace("inred", "in red")
    text = text.replace("ahealth", "a health")
    text = text.replace("theinformation", "the information")
    text = text.replace("tofast", "to fast")
    text = text.replace("completebloodcount", "complete blood count")

    return text.strip()

    # Fix only known spacing artifacts observed in MedlinePlus pages.
    replacements = {
        "oxygenfrom": "oxygen from",
        "partof": "part of",
        "bloodcell": "blood cell",
        "completeblood": "complete blood",
        "collectedinto": "collected into",
        "takea": "take a",
        "inred": "in red",
        "ahealth": "a health",
        "theinformation": "the information",
        "tofast": "to fast",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def save_document(source: dict, text: str):
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    output_file = KNOWLEDGE_DIR / f"{source['id']}.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"TITLE: {source['title']}\n")
        file.write(f"SOURCE: {source['organization']}\n")
        file.write(f"URL: {source['url']}\n")
        file.write(f"CATEGORY: {source['category']}\n")
        file.write("\n")
        file.write(text)

    return output_file


def ingest_sources():
    sources = load_sources()

    print(f"Found {len(sources)} sources.")

    for source in sources:
        print(f"\nDownloading: {source['title']}")

        try:
            html = download_page(source["url"])
            text = extract_text(html)

            output_file = save_document(source, text)

            print(f"Saved: {output_file}")
            print(f"Characters: {len(text):,}")

        except requests.RequestException as error:
            print(f"Download failed: {error}")


if __name__ == "__main__":
    ingest_sources()