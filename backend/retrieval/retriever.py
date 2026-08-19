import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
QDRANT_DIR = BASE_DIR / "data" / "qdrant"

COLLECTION_NAME = "medical_reports"


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")

if not JINA_API_KEY:
    raise ValueError("JINA_API_KEY is missing from .env")


# --------------------------------------------------
# Qdrant
# --------------------------------------------------

client = QdrantClient(
    path=str(QDRANT_DIR)
)


# --------------------------------------------------
# Jina Embedding
# --------------------------------------------------

def get_query_embedding(query: str):

    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-embeddings-v3",
            "input": [query],
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["data"][0]["embedding"]


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def search(query: str, top_k: int = 5):

    query_vector = get_query_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    query = "What does a high white blood cell count mean?"

    print()
    print("====================================")
    print("QUERY")
    print("====================================")
    print(query)

    results = search(query, top_k=5)

    print()
    print("====================================")
    print("RETRIEVED CHUNKS")
    print("====================================")

    for i, result in enumerate(results, start=1):

        payload = result.payload

        print()
        print(f"--- Result {i} ---")
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {payload['chunk_id']}")
        print(f"Title: {payload['title']}")
        print(f"Category: {payload['category']}")
        print()
        print(payload["text"][:1000])
        