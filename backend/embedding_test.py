import time

from fastembed import TextEmbedding


print("Loading embedding model...")

start = time.perf_counter()

model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

load_time = time.perf_counter() - start

print(f"Model loaded in: {load_time:.2f}s")


text = "What does a high white blood cell count mean?"

start = time.perf_counter()

embedding = list(model.embed([text]))[0]

embedding_time = time.perf_counter() - start

print()
print("==============================")
print("EMBEDDING TEST")
print("==============================")
print(f"Embedding time: {embedding_time:.4f}s")
print(f"Dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")