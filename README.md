# RAG Document Q&A Agent

A Retrieval-Augmented Generation (RAG) agent built with **LangChain**, **FAISS**, and **OpenAI**.

Upload documents, build a vector index, and ask grounded questions with source citations.

## Features
- Load `.txt`, `.md`, and `.pdf` documents
- Chunking with `RecursiveCharacterTextSplitter`
- FAISS vector store persistence
- Grounded LLM answers with source snippets
- CLI + Streamlit UI

## Setup
```bash
cd rag-document-qa-agent
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Run CLI
```bash
python rag_agent.py
```

## Run Streamlit UI
```bash
streamlit run app.py
```

## Project layout
```
docs/           sample / uploaded documents
rag_agent.py    core RAG pipeline
app.py          Streamlit frontend
vectorstore/    saved FAISS index (generated)
```

## How it works
1. Documents are split into overlapping chunks
2. Chunks are embedded and stored in FAISS
3. A user question retrieves top-k similar chunks
4. The LLM answers using only that context

## Sample Outputs
![Pipeline](outputs/pipeline_overview.png)
![Sample Q&A](outputs/sample_qa_output.png)
