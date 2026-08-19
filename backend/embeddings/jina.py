import os
import requests

from dotenv import load_dotenv


load_dotenv()


JINA_API_URL = "https://api.jina.ai/v1/embeddings"

JINA_API_KEY = os.getenv("JINA_API_KEY")

MODEL_NAME = "jina-embeddings-v5-text-small"


def embed_texts(texts: list[str]) -> list[list[float]]:

    if not JINA_API_KEY:
        raise RuntimeError(
            "JINA_API_KEY is not set in .env"
        )

    response = requests.post(
        JINA_API_URL,
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "input": texts,
            "embedding_type": "float",
            "normalized": True,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    embeddings = [
        item["embedding"]
        for item in data["data"]
    ]

    return embeddings