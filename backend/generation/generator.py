import os

from dotenv import load_dotenv
from groq import Groq

from backend.retrieval.retriever import search


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")


# --------------------------------------------------
# Groq Client
# --------------------------------------------------

client = Groq(
    api_key=GROQ_API_KEY
)


MODEL_NAME = "openai/gpt-oss-20b"


# --------------------------------------------------
# Build Context
# --------------------------------------------------

def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        payload = result.payload

        context_parts.append(
            f"""
SOURCE {i}

Title: {payload["title"]}
Source: {payload["source"]}
Category: {payload["category"]}
URL: {payload["url"]}

Content:
{payload["text"]}
""".strip()
        )

    return "\n\n-------------------------\n\n".join(
        context_parts
    )


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

def generate_answer(query: str, top_k: int = 5):

    # 1. Retrieve relevant chunks
    results = search(
        query,
        top_k=top_k
    )

    if not results:
        return {
            "answer": "I could not find relevant information in the medical knowledge base.",
            "sources": []
        }

    # 2. Build context
    context = build_context(results)

    # 3. System instructions
    system_prompt = """
You are MedicalReportCompanion, a medical information assistant.

Your job is to explain laboratory test information using ONLY
the provided medical knowledge context.

Rules:

1. Use only information supported by the provided context.
2. Do not invent medical facts.
3. Do not diagnose the user.
4. Do not recommend personalized treatment or medication.
5. If the context does not contain enough information, clearly say so.
6. Explain medical terminology in simple language.
7. Mention when abnormal results can have multiple possible explanations.
8. Encourage the user to discuss concerning or abnormal results with
   a qualified healthcare professional.
9. Keep the answer concise and easy to understand.
10. Do not claim certainty when the source does not provide certainty.

Answer the user's question directly.
"""

    # 4. User prompt
    user_prompt = f"""
Medical knowledge context:

{context}

User question:

{query}

Based strictly on the context above, provide a clear answer.
"""

    # 5. Call LLM
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=500,
    )

    answer = completion.choices[0].message.content

    # 6. Prepare source information
    sources = []

    for result in results:

        payload = result.payload

        sources.append({
            "chunk_id": payload["chunk_id"],
            "title": payload["title"],
            "source": payload["source"],
            "category": payload["category"],
            "url": payload["url"],
            "score": result.score,
        })

    return {
        "answer": answer,
        "sources": sources
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    query = "What does a high white blood cell count mean?"

    print()
    print("====================================")
    print("MEDICAL RAG")
    print("====================================")

    print(f"Question: {query}")

    result = generate_answer(
        query,
        top_k=5
    )

    print()
    print("====================================")
    print("ANSWER")
    print("====================================")

    print(result["answer"])

    print()
    print("====================================")
    print("SOURCES")
    print("====================================")

    for source in result["sources"]:

        print(
            f"- {source['title']} "
            f"({source['source']}) "
            f"[score={source['score']:.4f}]"
        )

        print(f"  {source['url']}")