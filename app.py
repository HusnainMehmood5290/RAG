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
from rag.pipeline import AnswerResult, answer_question

logger = logging.getLogger("app")


def _print_answer(result: AnswerResult) -> None:
    print("\n=== Answer ===")
    print(result.answer)
    sources = sorted({d.metadata.get("source", "unknown") for d in result.context})
    if sources:
        print("\n--- Sources ---")
        for s in sources:
            print(f"  • {s}")
    if result.truncated:
        print("\n(note: some retrieved documents were omitted to fit the context budget)")


def _history_from(messages: list[dict]) -> list[tuple[str, str]]:
    """Convert stored chat turns into (role, text) pairs for query condensation."""
    return [(m["role"], m["content"]) for m in messages]


def interactive_loop() -> None:
    print("Parent-Document RAG chatbot. Type 'q' to quit, 'clear' to reset history.")
    history: list[dict] = []
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
        if query.lower() == "clear":
            history.clear()
            print("(conversation history cleared)")
            continue
        try:
            result = answer_question(query, history=_history_from(history))
            _print_answer(result)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": result.answer})
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
    except Exception as exc:
        logger.exception("Unhandled error.")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
