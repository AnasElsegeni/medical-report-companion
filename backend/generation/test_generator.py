from backend.generation.generator import generate_answer


questions = [
    "What does a high white blood cell count mean?",
    "What does a low white blood cell count mean?",
    "What is a CBC?",
    "What does a high creatinine level mean?",
    "What is HbA1c used for?",
    "What does TSH measure?",
    "What is cholesterol?",
    "What does a liver function test measure?",
]


for i, question in enumerate(questions, start=1):

    print("\n" + "=" * 60)
    print(f"TEST {i}")
    print("=" * 60)

    print(f"\nQUESTION:\n{question}")

    try:
        result = generate_answer(question)

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")

        for source in result["sources"]:
            print(
                f"- {source['title']} | "
                f"{source['source']} | "
                f"score={source['score']:.4f}"
            )
            print(f"  {source['url']}")

    except Exception as e:
        print(f"\nERROR: {e}")