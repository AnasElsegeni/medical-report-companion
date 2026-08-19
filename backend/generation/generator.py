
import os
import re

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
# Configuration
# ==================================================

MIN_SCORE = 0.15
TOP_K = 5
MAX_TOKENS = 700


# ==================================================
# Arabic Detection
# ==================================================

def contains_arabic(text: str) -> bool:
    """
    Detect whether the user's question contains
    Arabic characters.
    """

    return bool(
        re.search(
            r"[\u0600-\u06FF]",
            text
        )
    )


# ==================================================
# Fallback Responses
# ==================================================

def get_fallback_response(
    user_is_arabic: bool
):

    if user_is_arabic:

        return (
            "لم أجد معلومات طبية موثوقة وكافية "
            "في المصادر المتاحة للإجابة عن هذا السؤال."
        )

    return (
        "I could not find enough reliable "
        "information in the available medical "
        "sources to answer this question."
    )


# ==================================================
# Arabic → English Translation
# ==================================================

def translate_arabic_to_english(
    question: str
) -> str:

    try:

        completion = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": """
You are a medical query translator.

Translate the user's Arabic medical question
into clear English for medical information retrieval.

Rules:

- Preserve the exact medical meaning.
- Preserve medical terms and abbreviations.
- Do not answer the question.
- Do not add medical information.
- Do not remove important details.
- Return ONLY the English translation.
"""
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],

            temperature=0,

            max_tokens=150,
        )

        translated = (
            completion
            .choices[0]
            .message
            .content
            .strip()
        )

        if translated:

            return translated

    except Exception as error:

        print(
            f"Translation error: {error}"
        )

    # Fallback
    return question


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
# Out-of-Domain Patterns
# ==================================================

OUT_OF_DOMAIN_PATTERNS = [

    # English

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

    # Arabic

    "علاج",
    "ما علاج",
    "ما هو علاج",
    "ما هي أعراض",
    "ما اعراض",
    "أعراض",
    "اعراض",

    "هل يمكنني تناول",
    "هل يمكنني اخذ",
    "هل يمكنني أخذ",

    "هل أستطيع تناول",
    "هل استطيع تناول",

    "دواء",
    "أدوية",
    "ادوية",

    "تشخيص",
    "شخصني",

    "ضغط الدم الطبيعي",
    "الضغط الطبيعي",
]


# ==================================================
# Out-of-Domain Detection
# ==================================================

def is_out_of_domain_question(
    question: str
) -> bool:

    question_lower = (
        question
        .lower()
        .strip()
    )

    for pattern in OUT_OF_DOMAIN_PATTERNS:

        if pattern in question_lower:

            return True

    return False


# ==================================================
# Relevance Gate
# ==================================================

def passes_relevance_gate(
    question: str,
    results
) -> bool:

    if not results:

        return False

    question_lower = (
        question
        .lower()
    )

    top_result = results[0]

    payload = (
        top_result.payload
        or {}
    )

    title = (
        payload
        .get("title", "")
        .lower()
    )

    category = (
        payload
        .get("category", "")
        .lower()
    )

    text = (
        payload
        .get("text", "")
        .lower()
    )

    matched_terms = []

    for term in MEDICAL_QUERY_TERMS:

        if term in question_lower:

            matched_terms.append(term)

    # --------------------------------------------------
    # Known medical topic
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
    # --------------------------------------------------

    return top_result.score >= 0.30


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

        payload = (
            result.payload
            or {}
        )

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
    # 0. Language Detection
    # ==================================================

    user_is_arabic = contains_arabic(
        question
    )

    print(
        "Language: "
        + (
            "Arabic"
            if user_is_arabic
            else "English"
        )
    )


    # ==================================================
    # 1. Explicit Out-of-Domain Check
    # ==================================================

    if is_out_of_domain_question(
        question
    ):

        print(
            "Decision: OUT OF DOMAIN"
        )

        return {

            "answer": get_fallback_response(
                user_is_arabic
            ),

            "sources": [],
        }


    # ==================================================
    # 2. Prepare Retrieval Query
    # ==================================================

    retrieval_question = question

    if user_is_arabic:

        retrieval_question = (
            translate_arabic_to_english(
                question
            )
        )

        print(
            f"Retrieval query: "
            f"{retrieval_question}"
        )


    # ==================================================
    # 3. Retrieval
    # ==================================================

    results, timing = search(

        query=retrieval_question,

        top_k=top_k
    )


    # ==================================================
    # 4. Debug Information
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
    # 5. Relevance Gate
    # ==================================================

    if not passes_relevance_gate(

        retrieval_question,

        results
    ):

        print(
            "Decision: OUT OF DOMAIN / "
            "INSUFFICIENT EVIDENCE"
        )

        return {

            "answer": get_fallback_response(
                user_is_arabic
            ),

            "sources": [],
        }


    # ==================================================
    # 6. Build Context
    # ==================================================

    context, relevant_results = (
        build_context(
            results,
            min_score=MIN_SCORE
        )
    )


    # ==================================================
    # 7. Evidence Check
    # ==================================================

    if not context:

        print(
            "Decision: NO SUFFICIENT EVIDENCE"
        )

        return {

            "answer": get_fallback_response(
                user_is_arabic
            ),

            "sources": [],
        }


    print(
        "Decision: USE CONTEXT "
        f"({len(relevant_results)} chunks)"
    )


    # ==================================================
    # 8. Language Instruction
    # ==================================================

    if user_is_arabic:

        language_instruction = """
Answer in Arabic.

Use clear, simple Arabic that is easy
for a non-specialist to understand.

Keep important medical terminology
accurate. When useful, include the
English medical term in parentheses.

Do NOT translate the source citations.
"""

    else:

        language_instruction = """
Answer in English using clear,
simple medical language.
"""


    # ==================================================
    # 9. System Prompt
    # ==================================================

    system_prompt = f"""
You are a medical information assistant.

Your job is to explain medical information
clearly and safely.

IMPORTANT RULES:

1. Answer ONLY using the provided CONTEXT.

2. Do not use outside knowledge.

3. Do not invent facts, numbers, diagnoses,
   treatments, or sources.

4. Do not diagnose the user.

5. Do not prescribe medications or treatments.

6. If the context does not contain enough
   information, say so.

7. Explain medical concepts in simple language.

8. Mention uncertainty when appropriate.

9. Do not assume that an abnormal laboratory
   result automatically means a disease.

10. Encourage consultation with a healthcare
    professional when interpretation depends
    on the patient's individual situation.

11. Never invent information that is not
    present in the CONTEXT.

12. The CONTEXT is the only source of
    medical information you are allowed
    to use.

13. {language_instruction}
"""


    # ==================================================
    # 10. User Prompt
    # ==================================================

    user_prompt = f"""
QUESTION:

{question}


CONTEXT:

{context}


{language_instruction}

Using ONLY the context above, answer
the question clearly and concisely.

Do not mention that you translated the
question.

If the context does not directly support
the answer, say that there is not enough
information in the provided medical sources.
"""


    # ==================================================
    # 11. Groq Request
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

        max_tokens=MAX_TOKENS,
    )


    answer = (

        completion
        .choices[0]
        .message
        .content
        .strip()
    )


    # ==================================================
    # 12. Sources
    # ==================================================

    sources = []

    seen = set()

    for result in relevant_results:

        payload = (
            result.payload
            or {}
        )

        url = payload.get(
            "url"
        )

        if not url:

            continue

        if url in seen:

            continue

        seen.add(url)

        sources.append({

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
        })


    # ==================================================
    # 13. Return
    # ==================================================

    return {

        "answer": answer,

        "sources": sources,

    }

