"""Interactive CLI for the RAG agent."""

from __future__ import annotations

import logging
import sys

from rag_qa.chain import ask
from rag_qa.config import get_settings
from rag_qa.store import build_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    print("RAG Document Q&A Agent")
    print(f"Provider: {settings.provider} · model: {settings.chat_model_name}")
    print(f"Docs: {settings.docs_dir}")
    print("Type a question (or 'quit' to exit). Rebuild index with: rebuild\n")

    try:
        settings.require_provider_ready()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "rebuild":
            try:
                build_vectorstore(settings=settings)
                print("Vector store rebuilt.\n")
            except Exception as exc:  # noqa: BLE001
                print(f"Error rebuilding index: {exc}\n")
            continue

        try:
            answer, sources = ask(question, settings=settings)
            print(f"\nAgent> {answer}\n")
            print("Retrieved chunks:")
            for i, doc in enumerate(sources, start=1):
                preview = doc.page_content[:120].replace("\n", " ")
                print(f"  {i}. {doc.metadata.get('source', '?')} — {preview}...")
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
