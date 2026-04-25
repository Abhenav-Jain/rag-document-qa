"""
app.py — RAGIntel
Render-ready version:
  - No local chroma_db dependency
  - In-memory Chroma (built from uploaded PDF each session)
  - Self-contained (no imports from main.py)
  - Query rewriting + Conversation memory + Logging
"""

import os
import json
import tempfile
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RAGIntel — Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

.stApp {
    background-color: #080b10;
    background-image:
        linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
}
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #1e2d3d !important;
}
[data-testid="stSidebar"] * { color: #c9d8e8; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 900px !important; }

.rag-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem; font-weight: 800;
    color: #ffffff; letter-spacing: -0.01em;
}
.rag-title span { color: #00d4ff; }
.rag-subtitle { font-size: 0.72rem; color: #5a7a94; letter-spacing: 0.06em; margin-bottom: 1.2rem; }

.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px;
    background: rgba(0,229,160,0.08); border: 1px solid rgba(0,229,160,0.25);
    border-radius: 20px; font-size: 0.65rem; color: #00e5a0;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1rem;
}
.section-label {
    font-size: 0.62rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: #364d62; margin: 16px 0 8px 0;
    padding-bottom: 6px; border-bottom: 1px solid #1e2d3d;
}
.stats-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0; }
.stat-card  { background: #131920; border: 1px solid #1e2d3d; border-radius: 8px; padding: 10px 12px; }
.stat-val   { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; color: #00d4ff; line-height: 1; }
.stat-label { font-size: 0.6rem; color: #364d62; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }

.msg-user {
    background: #00d4ff; color: #080b10;
    padding: 12px 16px; border-radius: 10px 2px 10px 10px;
    margin: 6px 0 6px 15%; font-size: 0.85rem; font-weight: 500; line-height: 1.6;
}
.msg-assistant {
    background: #131920; border: 1px solid #1e2d3d; color: #c9d8e8;
    padding: 14px 16px; border-radius: 2px 10px 10px 10px;
    margin: 6px 5% 6px 0; font-size: 0.85rem; line-height: 1.75;
}
.msg-header { font-size: 0.62rem; color: #364d62; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }
.rewrite-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; margin-bottom: 8px;
    background: rgba(0,212,255,0.06); border: 1px solid rgba(0,212,255,0.2);
    border-radius: 5px; font-size: 0.65rem; color: #00d4ff;
}
.sources-block { margin-top: 10px; padding-top: 10px; border-top: 1px solid #1e2d3d; }
.sources-label { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.1em; color: #364d62; margin-bottom: 6px; }
.source-chips  { display: flex; flex-wrap: wrap; gap: 5px; }
.source-chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 8px; background: #1a2230; border: 1px solid #263545;
    border-radius: 5px; font-size: 0.65rem; color: #5a7a94;
}
.not-found {
    margin-top: 8px; padding: 8px 12px;
    background: rgba(255,182,39,0.06); border: 1px solid rgba(255,182,39,0.2);
    border-radius: 6px; color: #ffb627; font-size: 0.72rem;
}
.chunk-box {
    background: #0d1117; border: 1px solid #1e2d3d; border-radius: 6px;
    padding: 10px 12px; font-size: 0.72rem; color: #5a7a94; line-height: 1.7; margin-bottom: 6px;
}
.stButton > button {
    background: #00d4ff !important; color: #080b10 !important; border: none !important;
    font-family: 'JetBrains Mono', monospace !important; font-weight: 600 !important;
    border-radius: 7px !important; letter-spacing: 0.04em !important;
}
.stButton > button:hover { background: #33ddff !important; }
div[data-testid="stChatInput"] > div {
    background: #0d1117 !important; border: 1px solid #263545 !important; border-radius: 10px !important;
}
div[data-testid="stChatInput"] textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important; color: #c9d8e8 !important; background: transparent !important;
}
.streamlit-expanderHeader {
    background: #131920 !important; color: #5a7a94 !important; font-size: 0.72rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid #1e2d3d !important; border-radius: 6px !important;
}
hr { border-color: #1e2d3d !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "messages":       [],
    "chat_history":   [],        # Conversation memory
    "query_count":    0,
    "total_latency":  0,
    "hit_count":      0,
    "retriever":      None,
    "llm":            None,
    "rag_ready":      False,
    "pdf_name":       None,      # Currently loaded PDF name
    "use_rewrite":    True,
    "use_memory":     True,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_llm():
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(model="mistral-small-2506")


@st.cache_resource
def get_embeddings():
    from langchain_mistralai import MistralAIEmbeddings
    return MistralAIEmbeddings(model="mistral-embed")


def build_vectorstore_from_pdf(uploaded_file):
    """
    PDF → chunks → in-memory Chroma vectorstore.
    No persist_directory — works on Render (no disk dependency).
    Returns (retriever, chunk_count).
    """
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Load PDF
        loader = PyMuPDFLoader(tmp_path)
        docs   = loader.load()

        # Tag source metadata
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        # Build IN-MEMORY vectorstore (no persist_directory)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            # No persist_directory → stays in RAM
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
        )

        return retriever, len(chunks)

    finally:
        os.unlink(tmp_path)


def get_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document." """
        ),
        (
            "human",
            """Chat History:
{history}

Context:
{context}

Question:
{question}"""
        ),
    ])


def rewrite_query(query: str, llm) -> str:
    """Rewrite user query to be more retrieval-friendly."""
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a query rewriting assistant.
Rewrite the user's question to be more specific and detailed for document search.
Return ONLY the rewritten query. No explanation."""
        ),
        ("human", "{query}"),
    ])
    try:
        result = llm.invoke(prompt.invoke({"query": query}))
        rewritten = result.content.strip()
        return rewritten if 0 < len(rewritten) < 500 else query
    except Exception:
        return query


def build_history(chat_history: list, max_turns: int = 3) -> str:
    """Last N conversation turns as a string for the prompt."""
    if not chat_history:
        return ""
    recent = chat_history[-(max_turns * 2):]
    lines  = [
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    ]
    return "\n".join(lines)


def log_query(query, rewritten, answer, sources, latency):
    """Append query log to logs.jsonl."""
    entry = {
        "timestamp":      datetime.now().isoformat(),
        "original_query": query,
        "rewritten_query":rewritten,
        "answer_preview": answer[:200],
        "sources":        sources,
        "latency_ms":     latency,
        "found_answer":   "could not find" not in answer.lower(),
    }
    try:
        with open("logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def safe_html(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>"))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="rag-title" style="font-size:1.4rem;">RAG<span>Intel</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-subtitle">RETRIEVAL-AUGMENTED GENERATION</div>', unsafe_allow_html=True)

    # System status
    if st.session_state.rag_ready:
        st.markdown(
            f'<div class="status-badge">● Ready — {st.session_state.pdf_name}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge" style="border-color:rgba(255,182,39,0.3);'
            'color:#ffb627;background:rgba(255,182,39,0.06);">◌ Upload a PDF to begin</div>',
            unsafe_allow_html=True,
        )

    # ── PDF Upload ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Upload Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded_file:
        # Only re-index if it's a new file
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner(f"Indexing {uploaded_file.name}…"):
                try:
                    retriever, n_chunks = build_vectorstore_from_pdf(uploaded_file)
                    st.session_state.retriever  = retriever
                    st.session_state.llm        = get_llm()
                    st.session_state.rag_ready  = True
                    st.session_state.pdf_name   = uploaded_file.name
                    st.session_state.messages   = []
                    st.session_state.chat_history = []
                    st.success(f"✅ {uploaded_file.name} — {n_chunks} chunks indexed")
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    # Show currently loaded PDF
    if st.session_state.pdf_name:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:7px 10px;
                    background:#131920;border:1px solid #1e2d3d;border-radius:6px;margin:4px 0;">
            <div style="background:rgba(255,77,109,0.15);color:#ff4d6d;font-size:9px;
                        font-weight:700;padding:2px 5px;border-radius:4px;">PDF</div>
            <div style="flex:1;font-size:11px;color:#c9d8e8;overflow:hidden;
                        text-overflow:ellipsis;white-space:nowrap;">{st.session_state.pdf_name}</div>
            <div style="width:6px;height:6px;border-radius:50%;background:#00e5a0;"></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature Toggles ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Features</div>', unsafe_allow_html=True)

    st.session_state.use_rewrite = st.toggle(
        "🔁 Query Rewriting",
        value=st.session_state.use_rewrite,
        help="Query ko LLM se rewrite karwao — better retrieval",
    )
    st.session_state.use_memory = st.toggle(
        "🧠 Conversation Memory",
        value=st.session_state.use_memory,
        help="Last 3 turns yaad rakhega — follow-up questions work karte hain",
    )

    if st.session_state.use_memory and st.session_state.chat_history:
        if st.button("🗑 Clear Memory", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Memory cleared!")

    # ── Session Stats ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Session Stats</div>', unsafe_allow_html=True)

    avg_lat  = (round(st.session_state.total_latency / st.session_state.query_count)
                if st.session_state.query_count else 0)
    hit_rate = (round((st.session_state.hit_count / st.session_state.query_count) * 100)
                if st.session_state.query_count else 100)

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card"><div class="stat-val">{st.session_state.query_count}</div><div class="stat-label">Queries</div></div>
        <div class="stat-card"><div class="stat-val">{avg_lat}ms</div><div class="stat-label">Avg Latency</div></div>
        <div class="stat-card"><div class="stat-val">{hit_rate}%</div><div class="stat-label">Hit Rate</div></div>
        <div class="stat-card"><div class="stat-val">4</div><div class="stat-label">Chunks/Q</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.session_state.query_count  = 0
        st.session_state.total_latency= 0
        st.session_state.hit_count    = 0
        st.rerun()

    # Logs download
    if os.path.exists("logs.jsonl"):
        st.markdown('<div class="section-label">Logs</div>', unsafe_allow_html=True)
        with open("logs.jsonl", "r", encoding="utf-8") as f:
            log_data = f.read()
        st.download_button(
            "📥 Download Logs", data=log_data,
            file_name="rag_logs.jsonl", mime="application/json",
            use_container_width=True,
        )

    st.markdown("""
    <div style="font-size:0.62rem;color:#364d62;margin-top:12px;line-height:1.8;">
        In-memory Chroma · No disk dependency<br>
        Splitter: Recursive · chunk=1000 · overlap=200<br>
        Model: mistral-small-2506
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="rag-title">Ask your <span>documents</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="rag-subtitle">Upload a PDF → Ask anything → Get precise answers</div>',
    unsafe_allow_html=True,
)

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    if not st.session_state.rag_ready:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#364d62;font-size:0.85rem;line-height:2.4;">
            📄 &nbsp; <strong style="color:#c9d8e8;">Upload a PDF</strong> in the sidebar to get started<br>
            <span style="font-size:0.72rem;">Your document will be indexed and ready to chat with instantly</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:0.65rem;color:#364d62;text-align:center;'
            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;margin-top:20px;">Try asking</div>',
            unsafe_allow_html=True,
        )
        _chips = [
            "Summarize the main themes",
            "What are the key concepts?",
            "Explain the introduction",
            "List important definitions",
        ]
        _cols = st.columns(len(_chips))
        for _i, _chip in enumerate(_chips):
            with _cols[_i]:
                if st.button(_chip, key=f"chip_{_i}", use_container_width=True):
                    st.session_state["prefill"] = _chip
                    st.rerun()

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user">{safe_html(msg["content"])}</div>',
            unsafe_allow_html=True,
        )
    else:
        is_not_found = "could not find" in msg["content"].lower()

        rewrite_html = ""
        if msg.get("rewritten_query") and msg["rewritten_query"] != msg.get("original_query"):
            rewrite_html = (
                f'<div class="rewrite-badge">🔁 {safe_html(msg["rewritten_query"])}</div><br>'
            )

        source_html = ""
        if msg.get("sources"):
            chips = "".join(f'<span class="source-chip">● {s}</span>' for s in msg["sources"])
            source_html = (
                '<div class="sources-block"><div class="sources-label">Retrieved Sources</div>'
                f'<div class="source-chips">{chips}</div></div>'
            )

        not_found_html = (
            '<div class="not-found">⚠ No relevant content found in the loaded document.</div>'
            if is_not_found else ""
        )

        latency_ms = msg.get("latency", 0)
        meta_str = f' &nbsp;·&nbsp; <span style="color:#00d4ff;">{latency_ms}ms</span>' if latency_ms else ""
        if msg.get("used_memory"):
            meta_str += ' &nbsp;·&nbsp; <span style="color:#00e5a0;">🧠</span>'

        st.markdown(f"""
        <div class="msg-assistant">
            <div class="msg-header">AI · RAGIntel{meta_str}</div>
            {rewrite_html}
            <div>{safe_html(msg["content"])}</div>
            {not_found_html}
            {source_html}
        </div>
        """, unsafe_allow_html=True)

        if msg.get("chunks"):
            with st.expander(f"📦 Retrieved chunks ({len(msg['chunks'])})"):
                for _i, _chunk in enumerate(msg["chunks"]):
                    st.markdown(f"**Chunk #{_i + 1}**")
                    st.markdown(
                        f'<div class="chunk-box">{safe_html(_chunk[:500])}{"…" if len(_chunk) > 500 else ""}</div>',
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT
# ══════════════════════════════════════════════════════════════════════════════
prefill = st.session_state.pop("prefill", "")
query   = st.chat_input("Ask a question about your document…") or prefill

if query:
    if not st.session_state.rag_ready:
        st.warning("📄 Please upload a PDF first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Retrieving context & generating answer…"):
            t0 = time.time()
            try:
                llm = st.session_state.llm

                # ── Query Rewriting ────────────────────────────────────────────
                rewritten = rewrite_query(query, llm) if st.session_state.use_rewrite else query

                # ── Memory ─────────────────────────────────────────────────────
                history_text = build_history(st.session_state.chat_history) if st.session_state.use_memory else ""

                # ── Retrieval ──────────────────────────────────────────────────
                docs    = st.session_state.retriever.invoke(rewritten)
                context = "\n\n".join([d.page_content for d in docs])

                # ── LLM ────────────────────────────────────────────────────────
                prompt  = get_prompt()
                fmt     = prompt.invoke({"context": context, "question": query, "history": history_text})
                result  = llm.invoke(fmt)

                latency = round((time.time() - t0) * 1000)
                answer  = result.content.strip()
                sources = list({
                    d.metadata.get("source", "document").replace("\\", "/").split("/")[-1]
                    for d in docs
                })
                chunks  = [d.page_content for d in docs]

                # ── Update memory ──────────────────────────────────────────────
                if st.session_state.use_memory:
                    st.session_state.chat_history.append({"role": "user",      "content": query})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    # Keep last 10 turns max
                    if len(st.session_state.chat_history) > 20:
                        st.session_state.chat_history = st.session_state.chat_history[-20:]

                # ── Log ────────────────────────────────────────────────────────
                log_query(query, rewritten, answer, sources, latency)

                # ── Save message ───────────────────────────────────────────────
                st.session_state.messages.append({
                    "role":            "assistant",
                    "content":         answer,
                    "sources":         sources,
                    "chunks":          chunks,
                    "latency":         latency,
                    "original_query":  query,
                    "rewritten_query": rewritten,
                    "used_memory":     st.session_state.use_memory,
                })
                st.session_state.query_count   += 1
                st.session_state.total_latency += latency
                if "could not find" not in answer.lower():
                    st.session_state.hit_count += 1

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", "content": f"⚠ Error: {e}",
                    "sources": [], "chunks": [], "latency": 0,
                    "original_query": query, "rewritten_query": query, "used_memory": False,
                })

        st.rerun()