from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
)


BASE_DIR = Path(__file__).resolve().parent.parent

QDRANT_PATH = BASE_DIR / "data" / "qdrant"

COLLECTION_NAME = "medical_reports"


def get_client():

    QDRANT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    return client


def create_collection(
    client,
    vector_size: int
):

    existing_collections = [
        collection.name
        for collection in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing_collections:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created collection: "
        f"{COLLECTION_NAME}"
    )
    