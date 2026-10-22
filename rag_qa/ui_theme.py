"""Visual theme for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

CSS = """
:root {
  --ink: #101828;
  --muted: #667085;
  --paper: #FBFCFB;
  --mist: #F2F5F3;
  --line: #E4E9E6;
  --teal: #0E7C74;
  --teal-deep: #0A5C56;
  --glow: rgba(14, 124, 116, 0.16);
  --sans: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --serif: Georgia, "Times New Roman", serif;
}

html, body, [class*="css"]  {
  font-family: var(--sans) !important;
}

/* Hide Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { display: none !important; }

.stApp {
  background:
    linear-gradient(180deg, #FAFCFB 0%, #F3F6F4 40%, #EEF2F0 100%);
}

.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    radial-gradient(circle at 18% 12%, var(--glow), transparent 28%),
    radial-gradient(circle at 88% 8%, rgba(16, 24, 40, 0.04), transparent 24%);
  z-index: 0;
}

.block-container {
  position: relative;
  z-index: 1;
  padding-top: 2.2rem !important;
  padding-bottom: 6rem !important;
  max-width: 860px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #0C1419 !important;
  border-right: 1px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] > div {
  padding-top: 1.4rem;
}

section[data-testid="stSidebar"] * {
  color: #E8EEF2 !important;
}

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label {
  color: #8B9AA6 !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.03);
  border: 1px dashed rgba(255,255,255,0.16);
  border-radius: 16px;
  padding: 1rem 0.85rem;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
  padding: 0 !important;
}

section[data-testid="stSidebar"] .stButton > button {
  height: 2.6rem;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.1);
  background: transparent;
  color: #F4F7F8 !important;
  font-weight: 600;
  letter-spacing: 0.01em;
  transition: 0.2s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
  border-color: rgba(45, 212, 191, 0.5);
  background: rgba(14, 124, 116, 0.22);
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #14A89C 0%, #0E7C74 100%);
  border: none;
  box-shadow: 0 12px 28px rgba(14, 124, 116, 0.35);
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
}

.side-kicker {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #5EEAD4 !important;
  margin: 0 0 0.35rem;
}

.side-brand {
  font-family: var(--serif) !important;
  font-size: 2rem !important;
  line-height: 1 !important;
  color: #F8FAFC !important;
  margin: 0 0 0.35rem !important;
}

.side-sub {
  color: #93A4B0 !important;
  font-size: 0.9rem !important;
  line-height: 1.45 !important;
  margin: 0 0 1.35rem !important;
}

.side-label {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #7C8C98 !important;
  margin: 0 0 0.55rem !important;
}

.file-card {
  margin: 0.85rem 0 1rem;
  padding: 0.95rem 1rem;
  border-radius: 14px;
  background: linear-gradient(145deg, rgba(20,168,156,0.16), rgba(255,255,255,0.03));
  border: 1px solid rgba(94, 234, 212, 0.22);
}

.file-card .label {
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #99F6E4 !important;
  margin-bottom: 0.3rem;
}

.file-card .name {
  font-weight: 600;
  color: #F0FDFA !important;
  word-break: break-word;
  font-size: 0.95rem;
}

.file-card .meta {
  margin-top: 0.25rem;
  color: #9FB0BB !important;
  font-size: 0.8rem;
}

.side-meta {
  margin-top: 1.1rem;
  padding-top: 0.95rem;
  border-top: 1px solid rgba(255,255,255,0.08);
  font-size: 0.78rem;
  color: #7E8F9B !important;
  line-height: 1.6;
}

.side-meta strong {
  color: #CBD5E1 !important;
  font-weight: 600;
}

/* Hero - full-bleed feel inside content */
.hero {
  margin: 0 0 1.5rem;
  padding: 0.2rem 0 0.4rem;
  animation: rise 0.55s ease both;
}

.hero-brand {
  font-family: var(--serif);
  font-size: 1.05rem;
  color: var(--teal-deep);
  margin: 0 0 0.85rem;
  letter-spacing: 0.02em;
}

.hero h1 {
  font-family: var(--serif);
  font-weight: 400;
  font-size: clamp(2.5rem, 5vw, 3.5rem);
  line-height: 1.05;
  color: var(--ink);
  margin: 0 0 0.85rem;
  letter-spacing: -0.02em;
}

.hero h1 em {
  font-style: italic;
  color: var(--teal);
}

.hero p {
  margin: 0;
  max-width: 34rem;
  color: var(--muted);
  font-size: 1.05rem;
  line-height: 1.55;
  font-weight: 400;
}

.status-line {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.65rem 1.1rem;
  margin: 0 0 1.75rem;
  padding: 0.75rem 0;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  animation: rise 0.7s ease both;
}

.status-item {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--muted);
}

.status-item .dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #98A2B3;
}

.status-item.ok { color: #067647; }
.status-item.ok .dot { background: #12B76A; }
.status-item.warn { color: #B54708; }
.status-item.warn .dot { background: #F79009; }
.status-item.live .dot { background: var(--teal); }

.empty {
  padding: 1.6rem 0 0.5rem;
  animation: rise 0.8s ease both;
}

.empty h2 {
  font-family: var(--serif);
  font-size: 1.75rem;
  font-weight: 400;
  color: var(--ink);
  margin: 0 0 0.5rem;
}

.empty p {
  margin: 0 0 1.4rem;
  color: var(--muted);
  max-width: 32rem;
  line-height: 1.6;
}

.flow {
  display: grid;
  grid-template-columns: 1fr auto 1fr auto 1fr;
  gap: 0.65rem;
  align-items: stretch;
}

.flow-card {
  padding: 1rem 0.15rem 0.2rem;
}

.flow-card .n {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--teal);
  margin-bottom: 0.4rem;
}

.flow-card strong {
  display: block;
  color: var(--ink);
  font-size: 1rem;
  margin-bottom: 0.25rem;
}

.flow-card span {
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.45;
}

.flow-sep {
  width: 1px;
  background: var(--line);
  align-self: stretch;
  margin: 0.4rem 0;
}

div[data-testid="stChatMessage"] {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(16, 24, 40, 0.06);
  border-radius: 18px;
  padding: 0.45rem 0.65rem;
  margin-bottom: 0.7rem;
  backdrop-filter: blur(8px);
  animation: rise 0.35s ease both;
}

[data-testid="stChatInput"] {
  background: transparent !important;
}

[data-testid="stChatInput"] > div {
  border: 1px solid var(--line) !important;
  background: #FFFFFF !important;
  border-radius: 18px !important;
  box-shadow: 0 16px 40px rgba(16, 24, 40, 0.06);
}

[data-testid="stExpander"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #FFFFFF;
}

hr.soft {
  border: none;
  border-top: 1px solid rgba(255,255,255,0.08);
  margin: 1.1rem 0 1rem;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 820px) {
  .flow {
    grid-template-columns: 1fr;
  }
  .flow-sep { display: none; }
  .hero h1 { font-size: 2.3rem; }
}
"""


def inject_theme() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-brand">DocuMind</div>
          <h1>Ask your PDF.<br/><em>Get grounded answers.</em></h1>
          <p>
            One document. Local indexing. Cited replies - private to this session.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pills(
    *,
    has_pdf: bool,
    index_ready: bool,
    provider: str,
    device: str,
) -> None:
    pdf_class = "ok" if has_pdf else ""
    pdf_label = "PDF ready" if has_pdf else "Waiting for PDF"
    idx_class = "ok" if index_ready else ("warn" if has_pdf else "")
    idx_label = "Indexed" if index_ready else ("Needs indexing" if has_pdf else "Not indexed")
    st.markdown(
        f"""
        <div class="status-line">
          <div class="status-item {pdf_class}"><span class="dot"></span>{pdf_label}</div>
          <div class="status-item {idx_class}"><span class="dot"></span>{idx_label}</div>
          <div class="status-item live"><span class="dot"></span>{provider} · {device.upper()}</div>
          <div class="status-item"><span class="dot"></span>Clears on refresh</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty">
          <h2>Start with a single PDF</h2>
          <p>
            Upload in the sidebar, index once, then ask. Refreshing the page
            clears the file, index, and conversation.
          </p>
          <div class="flow">
            <div class="flow-card">
              <div class="n">01</div>
              <strong>Upload</strong>
              <span>Drop one PDF into the sidebar.</span>
            </div>
            <div class="flow-sep"></div>
            <div class="flow-card">
              <div class="n">02</div>
              <strong>Index</strong>
              <span>Build a fast local GPU index.</span>
            </div>
            <div class="flow-sep"></div>
            <div class="flow-card">
              <div class="n">03</div>
              <strong>Ask</strong>
              <span>Get answers with source citations.</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
