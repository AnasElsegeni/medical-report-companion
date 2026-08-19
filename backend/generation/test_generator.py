from backend.generation.generator import generate_answer


query = "What does a high white blood cell count mean?"


result = generate_answer(
    query,
    top_k=5
)


print()
print("====================================")
print("RAG TEST")
print("====================================")

print()
print("QUESTION:")
print(query)

print()
print("ANSWER:")
print(result["answer"])

print()
print("SOURCES:")

for source in result["sources"]:

    print(
        f"- {source['title']} "
        f"| {source['source']} "
        f"| score={source['score']:.4f}"
    )

    print(f"  {source['url']}")