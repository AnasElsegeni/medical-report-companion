import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")

url = "https://api.jina.ai/v1/embeddings"

headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "jina-embeddings-v3",
    "input": [
        "What does a high white blood cell count mean?"
    ],
}


for i in range(3):

    print(f"\nRequest {i + 1}")

    start = time.perf_counter()

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120,
    )

    elapsed = time.perf_counter() - start

    response.raise_for_status()

    data = response.json()

    embedding = data["data"][0]["embedding"]

    print(f"Time: {elapsed:.2f}s")
    print(f"Dimensions: {len(embedding)}")
  