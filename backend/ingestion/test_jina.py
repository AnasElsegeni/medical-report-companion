import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("JINA_API_KEY")

if not API_KEY:
    raise ValueError("JINA_API_KEY is missing from .env")


url = "https://api.jina.ai/v1/embeddings"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "jina-embeddings-v3",
    "input": [
        "What is a complete blood count?"
    ],
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=60,
)

print("Status:", response.status_code)

if response.ok:
    data = response.json()

    embedding = data["data"][0]["embedding"]

    print("Embedding generated successfully!")
    print("Embedding dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])

else:
    print("Jina API Error:")
    print(response.text)