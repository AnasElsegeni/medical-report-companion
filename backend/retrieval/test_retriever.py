from backend.retrieval.retriever import search


query = "What does a high white blood cell count mean?"

results = search(query, top_k=3)

print()
print("====================================")
print("RETRIEVAL TEST")
print("====================================")
print(f"Query: {query}")

for i, result in enumerate(results, start=1):

    payload = result.payload

    print()
    print(f"--- Result {i} ---")
    print(f"Score: {result.score:.4f}")
    print(f"Chunk ID: {payload['chunk_id']}")
    print(f"Category: {payload['category']}")
    print(f"Title: {payload['title']}")
    print()
    print(payload["text"])