import os

from dotenv import load_dotenv
from groq import Groq

from backend.retrieval.retriever import search


# ==================================================
# Environment
# ==================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from .env"
    )


# ==================================================
# Groq
# ==================================================

client = Groq(
    api_key=GROQ_API_KEY
)

MODEL_NAME = "openai/gpt-oss-20b"


# ==================================================
# Retrieval Configuration
# ==================================================

MIN_SCORE = 0.15
TOP_K = 5


# ==================================================
# Medical Query Terms
# ==================================================

MEDICAL_QUERY_TERMS = {
    "cbc",
    "white blood cell",
    "white blood cells",
    "wbc",
    "red blood cell",
    "red blood cells",
    "rbc",
    "hemoglobin",
    "hematocrit",
    "platelet",
    "platelets",
    "creatinine",
    "kidney",
    "hba1c",
    "a1c",
    "diabetes",
    "tsh",
    "thyroid",
    "cholesterol",
    "lipid",
    "liver",
    "liver function",
    "blood sugar",
}


# ==================================================
# Out-of-Domain Question Detection
# ==================================================

OUT_OF_DOMAIN_PATTERNS = [
    "normal blood pressure",
    "blood pressure for",
    "normal bp",
    "treatment for",
    "treatment of",
    "can i take",
    "should i take",
    "what medication",
    "which medication",
    "medication for",
    "symptoms of",
    "signs of",
    "diagnose",
    "diagnosis",
]


def is_out_of_domain_question(question: str) -> bool:

    question_lower = question.lower().strip()

    for pattern in OUT_OF_DOMAIN_PATTERNS:

        if pattern in question_lower:
            return True

    return False


# ==================================================
# Medical Term Detection
# ==================================================

def has_medical_term(question: str) -> bool:

    question_lower = question.lower()

    return any(
        term in question_lower
        for term in MEDICAL_QUERY_TERMS
    )


# ==================================================
# Relevance Gate
# ==================================================

def passes_relevance_gate(
    question: str,
    results
) -> bool:

    if not results:
        return False

    question_lower = question.lower()

    top_result = results[0]

    payload = top_result.payload or {}

    title = payload.get(
        "title",
        ""
    ).lower()

    category = payload.get(
        "category",
        ""
    ).lower()

    text = payload.get(
        "text",
        ""
    ).lower()

    # --------------------------------------------------
    # Check whether the question contains
    # a known medical topic
    # --------------------------------------------------

    matched_terms = []

    for term in MEDICAL_QUERY_TERMS:

        if term in question_lower:

            matched_terms.append(term)

    # --------------------------------------------------
    # If a known medical term exists,
    # require the retrieved document to
    # contain the same topic.
    # --------------------------------------------------

    if matched_terms:

        for term in matched_terms:

            if (
                term in title
                or term in category
                or term in text
            ):
                return True

        return False

    # --------------------------------------------------
    # Semantic fallback
    #
    # Only allow a relatively strong score.
    # --------------------------------------------------

    if top_result.score >= 0.30:

        return True

    return False


# ==================================================
# Build Context
# ==================================================

def build_context(
    results,
    min_score=MIN_SCORE
):

    relevant_results = [
        result
        for result in results
        if result.score >= min_score
    ]

    if not relevant_results:
        return "", []

    context_parts = []

    for i, result in enumerate(
        relevant_results,
        start=1
    ):

        payload = result.payload or {}

        context_parts.append(
            f"""
SOURCE {i}

Title: {payload.get("title", "Unknown")}
Source: {payload.get("source", "Unknown")}
Category: {payload.get("category", "Unknown")}

Content:
{payload.get("text", "")}
""".strip()
        )

    return (
        "\n\n".join(context_parts),
        relevant_results
    )


# ==================================================
# Generate Answer
# ==================================================

def generate_answer(
    question: str,
    top_k: int = TOP_K
):

    # ==================================================
    # 1. Explicit Out-of-Domain Check
    # ==================================================

    if is_out_of_domain_question(question):

        print(
            "Decision: OUT OF DOMAIN"
        )

        return {
            "answer": (
                "I could not find enough reliable "
                "information in the available medical "
                "sources to answer this question."
            ),
            "sources": [],
        }
    
    # ==================================================
    # 2. Retrieval
    # ==================================================

    results, timing = search(
        query=question,
        top_k=top_k
    )

    # ==================================================
    # 3. Debug Information
    # ==================================================

    print()
    print("RETRIEVAL:")
    print(
        f"Results: {len(results)}"
    )

    if results:

        print(
            f"Top score: "
            f"{results[0].score:.4f}"
        )

    # ==================================================
    # 4. Relevance Gate
    # ==================================================

    if not passes_relevance_gate(
        question,
        results
    ):

        print(
            "Decision: OUT OF DOMAIN / "
            "INSUFFICIENT EVIDENCE"
        )

        return {
            "answer": (
                "I could not find enough reliable "
                "information in the available medical "
                "sources to answer this question."
            ),
            "sources": [],
        }

    # ==================================================
    # 5. Build Context
    # ==================================================

    context, relevant_results = build_context(
        results,
        min_score=MIN_SCORE
    )

    # ==================================================
    # 6. Evidence Check
    # ==================================================

    if not context:

        print(
            "Decision: NO SUFFICIENT EVIDENCE"
        )

        return {
            "answer": (
                "I could not find enough reliable "
                "information in the available medical "
                "sources to answer this question."
            ),
            "sources": [],
        }

    print(
        "Decision: USE CONTEXT "
        f"({len(relevant_results)} chunks)"
    )

    # ==================================================
    # 7. System Prompt
    # ==================================================

    system_prompt = """
You are a medical information assistant.

Your job is to explain medical information clearly
and safely.

IMPORTANT RULES:

1. Answer ONLY using the provided CONTEXT.
2. Do not use outside knowledge.
3. Do not invent facts, numbers, diagnoses,
   treatments, or sources.
4. Do not diagnose the user.
5. Do not prescribe medications or treatments.
6. If the context does not contain enough information,
   say so.
7. Explain medical concepts in simple language.
8. Mention uncertainty when appropriate.
9. Do not assume that an abnormal laboratory result
   automatically means a disease.
10. Encourage consultation with a healthcare professional
    when interpretation depends on the patient's
    individual situation.
"""

    # ==================================================
    # 8. User Prompt
    # ==================================================

    user_prompt = f"""
QUESTION:

{question}


CONTEXT:

{context}


Using ONLY the context above, answer the question
clearly and concisely.

If the context does not directly support the answer,
say that there is not enough information in the
provided medical sources.
"""

    # ==================================================
    # 9. Groq Request
    # ==================================================

    completion = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        temperature=0.1,

        max_tokens=500,
    )

    answer = (
        completion
        .choices[0]
        .message
        .content
    )

    # ==================================================
    # 10. Sources
    # ==================================================

    sources = []

    seen = set()

    for result in relevant_results:

        payload = result.payload or {}

        url = payload.get("url")

        if not url:
            continue

        if url in seen:
            continue

        seen.add(url)

        sources.append(
            {
                "title": payload.get(
                    "title",
                    "Unknown"
                ),

                "source": payload.get(
                    "source",
                    "Unknown"
                ),

                "url": url,

                "score": result.score,
            }
        )

    # ==================================================
    # 11. Return
    # ==================================================

    return {
        "answer": answer,
        "sources": sources,
    }