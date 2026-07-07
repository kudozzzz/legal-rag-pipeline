#Evaluation harness: precision@3 over manually written QA cases

import json
import sys
from pathlib import Path

from pipeline import RAGPipeline

_QUESTIONS_FILE = Path(__file__).parent / "sample_questions.json"
_GREEN = "\033[92m"
_RED = "\033[91m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def precision_at_3(results: dict, expected_document: str) -> bool:
    """True if expected_document appears in any of the top-3 retrieved sources."""
    return any(
        s["document"] == expected_document for s in results.get("sources", [])
    )


def run_evaluation(pdf_dir: str, index_path: str | None = None) -> None:
    cases = json.loads(_QUESTIONS_FILE.read_text())

    print(f"\nLoading pipeline from '{pdf_dir}'...")
    try:
        pipeline = RAGPipeline(pdf_dir=pdf_dir, index_path=index_path)
    except Exception as exc:
        print(f"Failed to initialise pipeline: {exc}", file=sys.stderr)
        sys.exit(1)

    correct = 0
    total = len(cases)

    print(f"\n{'─' * 60}")
    print(f"{'Q#':<4} {'PASS':<6} {'CONF':<7} QUESTION")
    print(f"{'─' * 60}")

    for i, case in enumerate(cases, start=1):
        question = case["question"]
        expected_doc = case["expected_document"]

        try:
            result = pipeline.query(question)
        except Exception as exc:
            print(f"{i:<4} {'ERROR':<6}        {question[:55]}")
            print(f"     → {exc}", file=sys.stderr)
            continue

        hit = precision_at_3(result, expected_doc)
        if hit:
            correct += 1

        status = f"{_GREEN}PASS{_RESET}" if hit else f"{_RED}FAIL{_RESET}"
        conf = f"{result['confidence']:.3f}"
        print(f"{i:<4} {status:<15} {conf:<7} {question[:52]}")

        if not hit:
            retrieved = [s["document"] for s in result.get("sources", [])]
            print(f"     expected: {expected_doc}  got: {retrieved}")

    print(f"{'─' * 60}")
    score = correct / total if total else 0.0
    label = f"{_BOLD}Precision@3: {correct}/{total} = {score:.2f}{_RESET}"
    print(f"\n{label}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline precision@3")
    parser.add_argument("pdf_dir", help="Directory containing PDF documents")
    parser.add_argument(
        "--index", default=None, help="Path prefix for saved FAISS index (optional)"
    )
    args = parser.parse_args()

    run_evaluation(pdf_dir=args.pdf_dir, index_path=args.index)
