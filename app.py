"""Streamlit UI - one PDF per browser session (cleared on refresh)."""

from __future__ import annotations

# Must run before Streamlit / FastEmbed / HuggingFace print progress.
import rag_qa.encoding_fix  # noqa: F401

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import streamlit as st

from rag_qa.config import get_settings
from rag_qa.providers import check_ollama_reachable, display_device
from rag_qa.store import index_exists
from rag_qa.ui_theme import inject_theme, render_empty_state, render_hero, render_status_pills

st.set_page_config(
    page_title="DocuMind RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
base_settings = get_settings()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "busy" not in st.session_state:
    st.session_state.busy = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "work_dir" not in st.session_state:
    st.session_state.work_dir = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "ollama_ok" not in st.session_state:
    st.session_state.ollama_ok = None


def _provider_ready() -> tuple[bool, str]:
    """Fast readiness check; cache Ollama ping for this browser session."""
    try:
        base_settings.require_provider_ready()
        if base_settings.provider == "ollama":
            if st.session_state.ollama_ok is True:
                return True, ""
            check_ollama_reachable(base_settings.ollama_base_url)
            st.session_state.ollama_ok = True
        return True, ""
    except (ValueError, ConnectionError) as exc:
        st.session_state.ollama_ok = False
        return False, str(exc)


def _session_settings():
    """Settings scoped to this browser session's single PDF workspace."""
    if not st.session_state.work_dir:
        return None
    work = Path(st.session_state.work_dir)
    return replace(
        base_settings,
        docs_dir=work / "docs",
        vector_dir=work / "vectorstore",
    )


def _reset_workspace() -> None:
    old = st.session_state.work_dir
    if old and Path(old).exists():
        shutil.rmtree(old, ignore_errors=True)
    st.session_state.work_dir = None
    st.session_state.index_ready = False
    st.session_state.messages = []


def _materialize_pdf() -> Path:
    """Write the session PDF into a fresh temp folder (one file only)."""
    if not st.session_state.pdf_bytes or not st.session_state.pdf_name:
        raise FileNotFoundError("No PDF uploaded in this session.")

    _reset_workspace()
    work = Path(tempfile.mkdtemp(prefix="rag_qa_session_"))
    docs = work / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    target = docs / Path(st.session_state.pdf_name).name
    target.write_bytes(st.session_state.pdf_bytes)
    st.session_state.work_dir = str(work)
    return target


embed_backend = base_settings.resolved_embedding_backend
# Avoid importing torch on first paint - use configured preference for labels.
embed_device = (
    display_device(base_settings.embedding_device)
    if embed_backend == "local"
    else embed_backend
)

ready, ready_msg = _provider_ready()
if not ready:
    render_hero()
    st.error(ready_msg)
    if base_settings.provider == "ollama":
        st.info(
            "Free local mode needs [Ollama](https://ollama.com) running, then:\n\n"
            "```\nollama pull llama3.2\n```"
        )
    st.stop()

with st.sidebar:
    st.markdown('<p class="side-kicker">Session workspace</p>', unsafe_allow_html=True)
    st.markdown('<p class="side-brand">DocuMind</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="side-sub">Private PDF Q&amp;A. One file at a time - '
        "cleared when you refresh.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<p class="side-label">Document</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key=f"pdf_uploader_{st.session_state.uploader_key}",
        help="Only one PDF is kept. A new upload replaces it. Refresh clears it.",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        data = uploaded.getvalue()
        name = Path(uploaded.name).name
        changed = (
            st.session_state.pdf_name != name
            or st.session_state.pdf_bytes != data
        )
        if changed:
            st.session_state.pdf_name = name
            st.session_state.pdf_bytes = data
            st.session_state.index_ready = False
            st.session_state.messages = []

    if st.session_state.pdf_name and st.session_state.pdf_bytes:
        size_kb = len(st.session_state.pdf_bytes) / 1024
        st.markdown(
            f"""
            <div class="file-card">
              <div class="label">Active document</div>
              <div class="name">{st.session_state.pdf_name}</div>
              <div class="meta">{size_kb:.0f} KB · one file only</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("No PDF in this session yet.")

    index_btn = st.button(
        "Index this PDF",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.busy or not st.session_state.pdf_bytes,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        remove = st.button(
            "Remove",
            use_container_width=True,
            disabled=st.session_state.busy or not st.session_state.pdf_bytes,
        )
    with col_b:
        clear = st.button(
            "Clear chat",
            use_container_width=True,
            disabled=st.session_state.busy,
        )

    if remove:
        st.session_state.pdf_name = None
        st.session_state.pdf_bytes = None
        _reset_workspace()
        st.session_state.uploader_key += 1
        st.rerun()

    if clear:
        st.session_state.messages = []
        st.rerun()

    engine = f"{base_settings.provider} / {base_settings.chat_model_name}"
    embed = f"{embed_backend} / {embed_device}"
    st.markdown(
        f"""
        <div class="side-meta">
          <div><strong>Engine</strong> · {engine}</div>
          <div><strong>Embed</strong> · {embed}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_hero()
session_settings = _session_settings()
has_pdf = bool(st.session_state.pdf_name and st.session_state.pdf_bytes)
index_ready_now = bool(
    has_pdf
    and st.session_state.index_ready
    and session_settings is not None
    and index_exists(settings=session_settings)
)
render_status_pills(
    has_pdf=has_pdf,
    index_ready=index_ready_now,
    provider=base_settings.provider,
    device=embed_device,
)

if index_btn:
    from rag_qa.store import build_vectorstore

    st.session_state.busy = True
    status = st.empty()
    bar = st.progress(0, text="Starting...")
    stats = {"chunks": 0}

    def on_progress(done: int, total: int, message: str) -> None:
        total = max(total, 1)
        if "Embedding" in message and "/" in message:
            stats["chunks"] = total
        bar.progress(min(done / total, 1.0), text=message)
        status.info(message)

    try:
        pdf_path = _materialize_pdf()
        session_settings = _session_settings()
        assert session_settings is not None
        status.info(
            f"Loading embed model (first time only), then indexing "
            f"**{pdf_path.name}**..."
        )
        build_vectorstore(settings=session_settings, progress=on_progress)
        st.session_state.index_ready = True
        bar.progress(1.0, text="Done")
        label = f" ({stats['chunks']} chunks)" if stats["chunks"] else ""
        status.success(f"Ready{label}. Ask questions about this PDF.")
        st.session_state.busy = False
    except Exception as exc:  # noqa: BLE001
        st.session_state.busy = False
        st.session_state.index_ready = False
        status.error(f"Indexing failed: {exc}")
        st.stop()

can_ask = bool(
    st.session_state.pdf_bytes
    and st.session_state.index_ready
    and session_settings is not None
    and index_exists(settings=session_settings)
)

if not st.session_state.messages:
    if not has_pdf:
        render_empty_state()
    elif not can_ask:
        st.markdown(
            """
            <div class="empty">
              <h2>PDF ready - index it next</h2>
              <p>
                Click <strong>Index this PDF</strong> in the sidebar.
                The first index loads the GPU model once; later runs are faster.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            for i, src in enumerate(message["sources"], start=1):
                with st.expander(f"Source {i} · {src['name']}"):
                    st.write(src["content"])

prompt = st.chat_input(
    "Ask anything about this PDF...",
    disabled=st.session_state.busy or not can_ask,
)

if prompt:
    from rag_qa.chain import ask

    session_settings = _session_settings()
    if not can_ask or session_settings is None:
        st.warning("Upload and index a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt, "sources": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your PDF and composing an answer..."):
                answer, sources = ask(prompt, settings=session_settings)
            st.markdown(answer)
            source_payload = []
            for i, doc in enumerate(sources, start=1):
                name = Path(doc.metadata.get("source", "unknown")).name
                source_payload.append({"name": name, "content": doc.page_content})
                with st.expander(f"Source {i} · {name}"):
                    st.write(doc.page_content)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": source_payload}
            )
        except Exception as exc:  # noqa: BLE001
            err = f"Could not answer that question: {exc}"
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "sources": None}
            )
