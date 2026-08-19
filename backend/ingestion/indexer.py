import json
import os
from pathlib import Path

from fastembed import TextEmbedding

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "data" / "chunks.json"
QDRANT_DIR = BASE_DIR / "data" / "qdrant"

COLLECTION_NAME = "medical_reports"


# --------------------------------------------------
# Environment
# --------------------------------------------------

# --------------------------------------------------
# Load chunks
# --------------------------------------------------

def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# Jina Embeddings
# --------------------------------------------------

def get_embeddings(model, texts):
    return list(model.embed(texts))

    response.raise_for_status()

    data = response.json()

    return [
        item["embedding"]
        for item in data["data"]
    ]


# --------------------------------------------------
# Qdrant
# --------------------------------------------------

def create_qdrant_collection(client):

    collections = client.get_collections()

    existing = [
        collection.name
        for collection in collections.collections
    ]

    if COLLECTION_NAME in existing:

        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        print("Old collection deleted.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created collection: {COLLECTION_NAME}"
    )


# --------------------------------------------------
# Index chunks
# --------------------------------------------------

def index_chunks(client, chunks, model):
  
    batch_size = 16
    total = len(chunks)

    for start in range(0, total, batch_size):

        batch = chunks[start:start + batch_size]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        print(
            f"Embedding chunks "
            f"{start + 1}-{min(start + batch_size, total)} "
            f"of {total}..."
        )

        embeddings = get_embeddings(model, texts)

        points = []

        for chunk, embedding in zip(batch, embeddings):

            point = PointStruct(
                id=start + len(points),
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk["id"],
                    "text": chunk["text"],
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "url": chunk["url"],
                    "category": chunk["category"],
                    "chunk_index": chunk["chunk_index"],
                    "file": chunk["file"],
                },
            )

            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        print(f"Indexed {len(points)} chunks.")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading chunks...")

    chunks = load_chunks()

    print(
        f"Found {len(chunks)} chunks."
    )
    
    print("Loading local embedding model...")

    model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    )

    print("Embedding model loaded.")

    print(
        "Opening persistent Qdrant database..."
    )

    QDRANT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = QdrantClient(
        path=str(QDRANT_DIR)
    )

    create_qdrant_collection(client)

    index_chunks(
    client,
    chunks,
    model,
   )

    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print()
    print("====================================")
    print("INDEXING COMPLETE")
    print("====================================")

    print(
        "Vectors stored:",
        collection_info.points_count
    )

    print(
        "Collection:",
        COLLECTION_NAME
    )

    print(
        "Database:",
        QDRANT_DIR
    )


if __name__ == "__main__":
    main()