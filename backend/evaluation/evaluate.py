import json
from pathlib import Path

from backend.retrieval.retriever import search
from backend.generation.generator import generate_answer

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    
    
# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(
    __file__
).resolve().parent.parent

TEST_CASES_FILE = (
    BASE_DIR
    / "evaluation"
    / "test_cases.json"
)


# ==================================================
# Load Test Cases
# ==================================================

def load_test_cases():

    with open(
        TEST_CASES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ==================================================
# Evaluate Retrieval
# ==================================================

def evaluate_retrieval(
    results,
    test_case
):

    expected_category = (
        test_case["expected_category"]
    )

    expected_source = (
        test_case["expected_source"]
    )


    # --------------------------------------------------
    # No results
    # --------------------------------------------------

    if not results:

        return False, 0.0


    # --------------------------------------------------
    # Top score
    # --------------------------------------------------

    top_score = max(
        result.score
        for result in results
    )


    # --------------------------------------------------
    # Out-of-domain
    # --------------------------------------------------

    # --------------------------------------------------
# Out-of-domain
# --------------------------------------------------

# --------------------------------------------------
# Out-of-domain
# --------------------------------------------------

    if (
    expected_category is None
    and expected_source is None
):

    # OOD is successful if the retrieved evidence
    # is not strong enough to support an answer.

        retrieval_ok = top_score < 0.25

        return (
        retrieval_ok,
        top_score
    )


    # --------------------------------------------------
    # Category Match
    # --------------------------------------------------

    category_match = any(

        result.payload.get(
            "category"
        )
        == expected_category

        for result in results
    )


    # --------------------------------------------------
    # Source Match
    # --------------------------------------------------

    source_match = any(

        result.payload.get(
            "source"
        )
        == expected_source

        for result in results
    )


    # --------------------------------------------------
    # Final Retrieval Result
    # --------------------------------------------------

    retrieval_ok = (
        category_match
        and source_match
    )


    return (
        retrieval_ok,
        top_score
    )


# ==================================================
# Main
# ==================================================

def main():

    test_cases = load_test_cases()


    print("=" * 60)
    print("RAG EVALUATION")
    print("=" * 60)


    total_tests = len(
        test_cases
    )


    print(
        f"\nTotal test cases: "
        f"{total_tests}"
    )


    # --------------------------------------------------
    # Counters
    # --------------------------------------------------

    retrieval_passed = 0
    answer_passed = 0

    in_domain_passed = 0
    in_domain_total = 0

    out_domain_rejected = 0
    out_domain_total = 0


    # ==================================================
    # Tests
    # ==================================================

    for index, test_case in enumerate(
        test_cases,
        start=1
    ):

        question = test_case[
            "question"
        ]

        should_answer = test_case[
            "should_answer"
        ]


        print()
        print("=" * 60)

        print(
            f"TEST {index}: "
            f"{test_case['id']}"
        )

        print("=" * 60)


        print("\nQUESTION:")
        print(question)


        try:

            # ==========================================
            # Retrieval
            # ==========================================

            results, timing = search(
                question,
                top_k=5
            )


            retrieval_ok, top_score = (
                evaluate_retrieval(
                    results,
                    test_case
                )
            )


            if retrieval_ok:

                retrieval_passed += 1


            # Count domain performance

            is_out_of_domain = (
                test_case[
                    "expected_category"
                ] is None
                and
                test_case[
                    "expected_source"
                ] is None
            )


            if is_out_of_domain:

                out_domain_total += 1

                if retrieval_ok:
                    out_domain_rejected += 1

            else:

                in_domain_total += 1

                if retrieval_ok:
                    in_domain_passed += 1


            print("\nRETRIEVAL:")


            print(
                f"Embedding time: "
                f"{timing['embedding']:.4f}s"
            )


            print(
                f"Qdrant time: "
                f"{timing['qdrant']:.4f}s"
            )


            print(
                f"Total retrieval time: "
                f"{timing['total']:.4f}s"
            )


            print(
                f"Top score: "
                f"{top_score:.4f}"
            )


            if results:

                first = (
                    results[0]
                    .payload
                )


                print(
                    f"Top source: "
                    f"{first.get('source')}"
                )


                print(
                    f"Top category: "
                    f"{first.get('category')}"
                )


            print(
                "Retrieval result: "
                f"{'PASS' if retrieval_ok else 'FAIL'}"
            )


            # ==========================================
            # Generation
            # ==========================================

            result = generate_answer(
                question
            )


            answer = result.get(
                "answer",
                ""
            )


            print("\nANSWER:")
            print(answer)


            # ==========================================
            # Answer Validation
            # ==========================================

            answer_is_empty = (
                not answer
                or not answer.strip()
            )


            if should_answer:

                # In-domain question
                answer_ok = (
                    not answer_is_empty
                )

            else:

                # Out-of-domain question
                # should trigger safe fallback.

                fallback_text = (
                    "I could not find enough reliable"
                )


                answer_ok = (
                    not answer_is_empty
                    and fallback_text.lower()
                    in answer.lower()
                )


            if answer_ok:

                answer_passed += 1


            print(
                "\nAnswer result: "
                f"{'PASS' if answer_ok else 'FAIL'}"
            )


            # ==========================================
            # Sources
            # ==========================================

            print("\nSOURCES:")


            sources = result.get(
                "sources",
                []
            )


            if not sources:

                print(
                    "- No sources"
                )


            else:

                for source in sources:

                    print(
                        f"- "
                        f"{source['title']} | "
                        f"{source['source']} | "
                        f"score="
                        f"{source['score']:.4f}"
                    )


        except Exception as error:

            print("\nERROR:")

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )


    # ==================================================
    # Final Summary
    # ==================================================

    retrieval_accuracy = (
        retrieval_passed
        / total_tests
        * 100
    )


    answer_accuracy = (
        answer_passed
        / total_tests
        * 100
    )


    if in_domain_total > 0:

        in_domain_accuracy = (
            in_domain_passed
            / in_domain_total
            * 100
        )

    else:

        in_domain_accuracy = 0


    if out_domain_total > 0:

        out_domain_accuracy = (
            out_domain_rejected
            / out_domain_total
            * 100
        )

    else:

        out_domain_accuracy = 0


    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)


    print(
        f"\nRetrieval passed: "
        f"{retrieval_passed}/"
        f"{total_tests}"
    )


    print(
        f"Answers generated: "
        f"{answer_passed}/"
        f"{total_tests}"
    )


    print(
        f"\nOverall retrieval accuracy: "
        f"{retrieval_accuracy:.1f}%"
    )


    print(
        f"In-domain retrieval: "
        f"{in_domain_passed}/"
        f"{in_domain_total} "
        f"({in_domain_accuracy:.1f}%)"
    )


    print(
        f"Out-of-domain rejection: "
        f"{out_domain_rejected}/"
        f"{out_domain_total} "
        f"({out_domain_accuracy:.1f}%)"
    )


    print(
        f"Answer pipeline success: "
        f"{answer_accuracy:.1f}%"
    )


    print(
        "\nEvaluation finished."
    )


# ==================================================
# Entry Point
# ==================================================

if __name__ == "__main__":

    main()