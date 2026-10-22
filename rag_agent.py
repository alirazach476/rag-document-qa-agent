"""RAG Document Q&A Agent — retrieve relevant chunks and answer with an LLM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
VECTOR_DIR = BASE_DIR / "vectorstore"


def load_documents(docs_path: Path = DOCS_DIR) -> List[Document]:
    docs_path.mkdir(parents=True, exist_ok=True)
    documents: List[Document] = []

    for pdf in docs_path.glob("**/*.pdf"):
        documents.extend(PyPDFLoader(str(pdf)).load())

    for txt in docs_path.glob("**/*.txt"):
        documents.extend(TextLoader(str(txt), encoding="utf-8").load())

    for md in docs_path.glob("**/*.md"):
        documents.extend(TextLoader(str(md), encoding="utf-8").load())

    return documents


def build_vectorstore(
    documents: Optional[List[Document]] = None,
    persist: bool = True,
) -> FAISS:
    docs = documents if documents is not None else load_documents()
    if not docs:
        raise FileNotFoundError(
            f"No documents found in {DOCS_DIR}. Add .txt, .md, or .pdf files."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    store = FAISS.from_documents(chunks, embeddings)

    if persist:
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(VECTOR_DIR))

    return store


def load_or_build_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    index_file = VECTOR_DIR / "index.faiss"
    if index_file.exists():
        return FAISS.load_local(
            str(VECTOR_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return build_vectorstore()


def format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def create_rag_chain(vectorstore: Optional[FAISS] = None):
    store = vectorstore or load_or_build_vectorstore()
    retriever = store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise RAG assistant. Answer ONLY using the provided context. "
                "If the answer is not in the context, say you do not know. "
                "Cite sources like [Source N] when relevant.\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.1,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(question: str) -> Tuple[str, List[Document]]:
    chain, retriever = create_rag_chain()
    sources = retriever.invoke(question)
    answer = chain.invoke(question)
    return answer, sources


def main() -> None:
    print("RAG Document Q&A Agent")
    print("Type a question (or 'quit' to exit). Rebuild index with: rebuild\n")

    while True:
        question = input("You> ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "rebuild":
            build_vectorstore()
            print("Vector store rebuilt.\n")
            continue
        try:
            answer, sources = ask(question)
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
