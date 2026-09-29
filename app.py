"""Interactive CLI for the Parent-Document RAG chatbot.

Usage:
    python app.py                 # start an interactive query loop
    python app.py --query "..."   # one-shot answer (for scripts/CI)
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import ConfigurationError
from logging_config import configure_logging
from rag.pipeline import answer_question

logger = logging.getLogger("app")


def _print_answer(result: dict) -> None:
    print("\n=== Answer ===")
    print(result["answer"])
    sources = sorted({d.metadata.get("source", "unknown") for d in result["context"]})
    if sources:
        print("\n--- Sources ---")
        for s in sources:
            print(f"  • {s}")


def interactive_loop() -> None:
    print("Parent-Document RAG chatbot. Type 'q' to quit.")
    while True:
        try:
            query = input("\nEnter query (press q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"q", "quit", "exit"}:
            break
        if not query:
            continue
        try:
            _print_answer(answer_question(query))
        except Exception as exc:  # keep the loop alive, but surface the error
            logger.exception("Failed to answer query.")
            print(f"ERROR: {exc}", file=sys.stderr)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Parent-Document RAG chatbot CLI")
    parser.add_argument("--query", help="Ask a single question and exit.")
    args = parser.parse_args()

    try:
        if args.query:
            _print_answer(answer_question(args.query))
        else:
            interactive_loop()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
