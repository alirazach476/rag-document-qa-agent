"""Streamlit UI for the RAG Document Q&A Agent."""

from pathlib import Path

import streamlit as st

from rag_agent import DOCS_DIR, ask, build_vectorstore

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")
st.title("RAG Document Q&A Agent")
st.caption("LangChain + FAISS + OpenAI — retrieve relevant chunks, then answer with an LLM.")

DOCS_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload .txt / .md / .pdf", type=["txt", "md", "pdf"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            (DOCS_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s) to docs/")

    if st.button("Rebuild vector index", use_container_width=True):
        with st.spinner("Indexing..."):
            build_vectorstore()
        st.success("Index rebuilt.")

    files = list(DOCS_DIR.glob("**/*"))
    st.write(f"Files in docs/: {len([f for f in files if f.is_file()])}")

question = st.text_input("Ask a question about your documents")
if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and generating..."):
        answer, sources = ask(question.strip())
    st.subheader("Answer")
    st.write(answer)
    st.subheader("Sources")
    for i, doc in enumerate(sources, start=1):
        with st.expander(f"Source {i}: {Path(doc.metadata.get('source', 'unknown')).name}"):
            st.write(doc.page_content)
