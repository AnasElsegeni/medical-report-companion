import json

from backend.embeddings.jina import embed_texts
from backend.vectorstore.qdrant_store import (
    get_client,
    create_collection,
    COLLECTION_NAME,
)


from qdrant_client.models import PointStruct


BASE_DIR = __import__("pathlib").Path(
    __file__
).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "data" / "chunks.json"


def load_chunks():

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "Creating embeddings using Jina..."
    )

    embeddings = embed_texts(texts)

    print(
        f"Received {len(embeddings)} embeddings."
    )

    vector_size = len(
        embeddings[0]
    )

    client = get_client()

    create_collection(
        client,
        vector_size
    )

    points = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        points.append(
            PointStruct(
                id=index,

                vector=embedding,

                payload={
                    "chunk_id": chunk["id"],
                    "text": chunk["text"],
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "url": chunk["url"],
                    "category": chunk["category"],
                    "file": chunk["file"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"Inserted {len(points)} vectors "
        f"into Qdrant."
    )

    print(
        f"Vector size: {vector_size}"
    )

    print(
        f"Database location: "
        f"backend/data/qdrant"
    )


if __name__ == "__main__":
    main()
    