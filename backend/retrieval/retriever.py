import re
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

QDRANT_DIR = BASE_DIR / "data" / "qdrant"

COLLECTION_NAME = "medical_reports"


# ==================================================
# Embedding Model
# ==================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading local embedding model...")

embedding_model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ==================================================
# Qdrant
# ==================================================

client = QdrantClient(
    path=str(QDRANT_DIR)
)


# ==================================================
# Query Embedding
# ==================================================

def get_query_embedding(query: str):

    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    return embedding.tolist()


# ==================================================
# Text Normalization
# ==================================================

def normalize_text(text: str):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==================================================
# Medical Synonyms / Abbreviations
# ==================================================

MEDICAL_TERMS = {

    "hba1c": [
        "hba1c",
        "hemoglobin a1c",
        "hemoglobin a1c test",
        "glycated hemoglobin",
        "a1c"
    ],

    "tsh": [
        "tsh",
        "thyroid stimulating hormone",
        "thyroid stimulating hormone test",
        "thyroid hormone"
    ],

    "cbc": [
        "cbc",
        "complete blood count",
        "complete blood cell count"
    ],

    "creatinine": [
        "creatinine",
        "creatinine test",
        "blood creatinine"
    ],

    "cholesterol": [
        "cholesterol",
        "cholesterol levels",
        "lipid",
        "lipid profile",
        "lipid panel"
    ],

    "liver": [
        "liver function",
        "liver function test",
        "liver tests",
        "hepatic function"
    ],

    "white blood cell": [
        "white blood cell",
        "white blood cells",
        "wbc",
        "white cell",
        "white cells"
    ],

    "red blood cell": [
        "red blood cell",
        "red blood cells",
        "rbc",
        "red cell",
        "red cells"
    ],

    "blood pressure": [
        "blood pressure",
        "systolic",
        "diastolic",
        "hypertension"
    ],
}


# ==================================================
# Extract Medical Terms
# ==================================================

def extract_medical_terms(query: str):

    normalized_query = normalize_text(query)

    detected_terms = []

    for canonical_term, variations in MEDICAL_TERMS.items():

        for variation in variations:

            variation = normalize_text(
                variation
            )

            if variation in normalized_query:

                detected_terms.append(
                    canonical_term
                )

                break

    return detected_terms


# ==================================================
# Calculate Keyword Score
# ==================================================

def calculate_keyword_score(
    query: str,
    text: str,
    payload
):

    normalized_query = normalize_text(
        query
    )

    normalized_text = normalize_text(
        text
    )

    normalized_title = normalize_text(
        payload.get("title", "")
    )

    normalized_category = normalize_text(
        payload.get("category", "")
    )

    detected_terms = extract_medical_terms(
        query
    )

    if not detected_terms:

        return 0.0

    score = 0.0

    for term in detected_terms:

        variations = MEDICAL_TERMS.get(
            term,
            []
        )

        for variation in variations:

            variation = normalize_text(
                variation
            )

            # Strong match in title
            if variation in normalized_title:

                score += 0.30

                break

            # Strong match in text
            if variation in normalized_text:

                score += 0.20

                break

            # Category match
            if variation in normalized_category:

                score += 0.15

                break

    return min(score, 0.50)


# ==================================================
# Retrieval
# ==================================================

def search(
    query: str,
    top_k: int = 5
):

    # --------------------------------------------------
    # 1. Embedding
    # --------------------------------------------------

    embedding_start = time.perf_counter()

    query_vector = get_query_embedding(
        query
    )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    # --------------------------------------------------
    # 2. Qdrant
    # --------------------------------------------------

    search_start = time.perf_counter()

    candidates = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=20,
        with_payload=True,
    ).points

    search_time = (
        time.perf_counter()
        - search_start
    )

    # --------------------------------------------------
    # 3. Hybrid Scoring
    # --------------------------------------------------

    scored_results = []

    for result in candidates:

        payload = result.payload

        text = payload.get(
            "text",
            ""
        )

        semantic_score = result.score

        keyword_score = calculate_keyword_score(
            query,
            text,
            payload
        )

        # ----------------------------------------------
        # Hybrid score
        # ----------------------------------------------

        final_score = (
            semantic_score
            + keyword_score
        )

        scored_results.append(
            (
                final_score,
                result
            )
        )

    # --------------------------------------------------
    # 4. Sort
    # --------------------------------------------------

    scored_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------
    # 5. Return top K
    # --------------------------------------------------

    results = [
        result
        for _, result in scored_results[:top_k]
    ]

    # --------------------------------------------------
    # 6. Timing
    # --------------------------------------------------

    timing = {

        "embedding": embedding_time,

        "qdrant": search_time,

        "total": (
            embedding_time
            + search_time
        ),
    }

    return results, timing


# ==================================================
# Test
# ==================================================

if __name__ == "__main__":

    query = (
        "What does TSH measure?"
    )

    print()
    print("=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)

    print(
        f"Query: {query}"
    )

    results, timing = search(
        query,
        top_k=5
    )

    print()
    print("=" * 60)
    print("RETRIEVAL TIMING")
    print("=" * 60)

    print(
        f"Local Embedding : "
        f"{timing['embedding']:.4f}s"
    )

    print(
        f"Qdrant Search   : "
        f"{timing['qdrant']:.4f}s"
    )

    print(
        f"Total Retrieval : "
        f"{timing['total']:.4f}s"
    )

    print()
    print("=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload

        print()
        print(
            f"--- Result {i} ---"
        )

        print(
            f"Score: "
            f"{result.score:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{payload.get('chunk_id')}"
        )

        print(
            f"Title: "
            f"{payload.get('title')}"
        )

        print(
            f"Category: "
            f"{payload.get('category')}"
        )

        print()

        print(
            payload.get(
                "text",
                ""
            )[:1000]
        )

    print()
    print("=" * 60)
    print("UNIQUE SOURCES")
    print("=" * 60)

    seen = set()

    for result in results:

        payload = result.payload

        url = payload.get(
            "url"
        )

        if url in seen:
            continue

        seen.add(url)

        print(
            f"{len(seen)}. "
            f"{payload.get('title')} | "
            f"{payload.get('source')}"
        )

        print(url)
        