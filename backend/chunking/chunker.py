import json
from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
CHUNKS_FILE = BASE_DIR / "data" / "chunks.json"

MAX_CHARS = 1200
OVERLAP_CHARS = 200


def parse_document(text: str):
    """
    Extract metadata from the document header.
    """

    lines = text.splitlines()

    metadata = {
        "title": "",
        "source": "",
        "url": "",
        "category": "",
    }

    content_start = 0

    for i, line in enumerate(lines):

        if line.startswith("TITLE:"):
            metadata["title"] = line.replace("TITLE:", "", 1).strip()

        elif line.startswith("SOURCE:"):
            metadata["source"] = line.replace("SOURCE:", "", 1).strip()

        elif line.startswith("URL:"):
            metadata["url"] = line.replace("URL:", "", 1).strip()

        elif line.startswith("CATEGORY:"):
            metadata["category"] = line.replace("CATEGORY:", "", 1).strip()

        elif i > 0 and line.strip() == "":
            content_start = i + 1
            break

    content = "\n".join(lines[content_start:]).strip()

    return metadata, content


def load_documents():

    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        metadata, content = parse_document(text)

        documents.append({
            "file": file_path.name,
            "metadata": metadata,
            "text": content,
        })

    return documents


def split_into_sections(text: str):

    blocks = re.split(
        r"\n\s*\n",
        text
    )

    sections = []

    current_section = []

    for block in blocks:

        block = block.strip()

        if not block:
            continue

        current_section.append(block)

        if block.endswith("?") and len(block) < 150:

            sections.append(
                "\n\n".join(current_section)
            )

            current_section = []

    if current_section:

        sections.append(
            "\n\n".join(current_section)
        )

    return sections


def create_chunks(text: str):

    sections = split_into_sections(text)

    chunks = []

    for section in sections:

        if len(section) <= MAX_CHARS:

            chunks.append(section)

            continue

        paragraphs = section.split("\n\n")

        current = ""

        for paragraph in paragraphs:

            if len(current) + len(paragraph) <= MAX_CHARS:

                current += paragraph + "\n\n"

            else:

                if current.strip():

                    chunks.append(
                        current.strip()
                    )

                overlap = current[-OVERLAP_CHARS:]

                current = (
                    overlap
                    + "\n\n"
                    + paragraph
                    + "\n\n"
                )

        if current.strip():

            chunks.append(
                current.strip()
            )

    return chunks


def main():

    documents = load_documents()

    all_chunks = []

    print(
        f"Found {len(documents)} documents."
    )

    for document in documents:

        chunks = create_chunks(
            document["text"]
        )

        metadata = document["metadata"]

        print(
            f"{document['file']}: "
            f"{len(chunks)} chunks"
        )

        for index, chunk in enumerate(chunks):

            chunk_id = (
                f"{metadata['category']}"
                f"_{index + 1:03d}"
            )

            all_chunks.append({

                "id": chunk_id,

                "text": chunk,

                "title": metadata["title"],

                "source": metadata["source"],

                "url": metadata["url"],

                "category": metadata["category"],

                "chunk_index": index,

                "file": document["file"],
            })

    CHUNKS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nTotal chunks: "
        f"{len(all_chunks)}"
    )

    print(
        f"Saved to: "
        f"{CHUNKS_FILE}"
    )


if __name__ == "__main__":
    main()