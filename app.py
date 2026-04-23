import streamlit as st
import time
import os
import tempfile
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

/* Title */
.rag-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem; font-weight: 800;
    color: #ffffff; letter-spacing: -0.01em; margin-bottom: 0;
}
.rag-title span { color: #00d4ff; }
.rag-subtitle {
    font-size: 0.72rem; color: #5a7a94;
    letter-spacing: 0.06em; margin-top: 2px; margin-bottom: 1.2rem;
}

/* Status badge */
.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px;
    background: rgba(0,229,160,0.08);
    border: 1px solid rgba(0,229,160,0.25);
    border-radius: 20px; font-size: 0.65rem; color: #00e5a0;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1rem;
}

/* Section label */
.section-label {
    font-size: 0.62rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: #364d62; margin: 16px 0 8px 0;
    padding-bottom: 6px; border-bottom: 1px solid #1e2d3d;
}

/* Stat cards */
.stats-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0; }
.stat-card  { background: #131920; border: 1px solid #1e2d3d; border-radius: 8px; padding: 10px 12px; }
.stat-val   { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; color: #00d4ff; line-height: 1; }
.stat-label { font-size: 0.6rem; color: #364d62; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }

/* Chat bubbles */
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
.msg-header {
    font-size: 0.62rem; color: #364d62;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;
}

/* Source chips */
.sources-block { margin-top: 10px; padding-top: 10px; border-top: 1px solid #1e2d3d; }
.sources-label { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.1em; color: #364d62; margin-bottom: 6px; }
.source-chips  { display: flex; flex-wrap: wrap; gap: 5px; }
.source-chip   {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 8px; background: #1a2230; border: 1px solid #263545;
    border-radius: 5px; font-size: 0.65rem; color: #5a7a94;
}

/* Not-found warning */
.not-found {
    margin-top: 8px; padding: 8px 12px;
    background: rgba(255,182,39,0.06);
    border: 1px solid rgba(255,182,39,0.2);
    border-radius: 6px; color: #ffb627; font-size: 0.72rem;
}

/* Chunk box */
.chunk-box {
    background: #0d1117; border: 1px solid #1e2d3d; border-radius: 6px;
    padding: 10px 12px; font-size: 0.72rem; color: #5a7a94;
    line-height: 1.7; margin-bottom: 6px;
}

/* Buttons */
.stButton > button {
    background: #00d4ff !important; color: #080b10 !important;
    border: none !important; font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important; border-radius: 7px !important;
    letter-spacing: 0.04em !important;
}
.stButton > button:hover { background: #33ddff !important; }

/* Chat input */
div[data-testid="stChatInput"] > div {
    background: #0d1117 !important;
    border: 1px solid #263545 !important;
    border-radius: 10px !important;
}
div[data-testid="stChatInput"] textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important; color: #c9d8e8 !important;
    background: transparent !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #131920 !important; color: #5a7a94 !important;
    font-size: 0.72rem !important; font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid #1e2d3d !important; border-radius: 6px !important;
}
hr { border-color: #1e2d3d !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "messages":        [],
    "query_count":     0,
    "total_latency":   0,
    "hit_count":       0,
    "rag_ready":       False,
    "retriever":       None,
    "llm":             None,
    "prompt":          None,
    "loaded_docs":     [],      # [{"name": str, "chunks": int}]
    "vectorstore":     None,
    "search_type_val": "mmr",
    "k_val":           4,
    "fetch_k_val":     10,
    "lambda_val":      0.5,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_embeddings():
    from langchain_mistralai import MistralAIEmbeddings
    return MistralAIEmbeddings(model="mistral-embed")


@st.cache_resource
def get_llm():
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(model="mistral-small-2506")


def get_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document." """
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        ),
    ])


def build_retriever(vectorstore, search_type, k, fetch_k, lambda_mult):
    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"]     = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def index_pdf(uploaded_file, vectorstore):
    """
    Save uploaded PDF to a temp file → chunk it → add to vectorstore.
    Returns number of chunks created.
    """
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        docs   = loader.load()

        # Tag each chunk with the original filename so sources show correctly
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)
        return len(chunks)
    finally:
        os.unlink(tmp_path)


def safe_html(text: str) -> str:
    """Escape text before injecting into HTML to prevent raw-tag rendering."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown('<div class="rag-title">RAG<span>Intel</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-subtitle">RETRIEVAL-AUGMENTED GENERATION</div>', unsafe_allow_html=True)

    if st.session_state.rag_ready:
        st.markdown('<div class="status-badge">● System Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="status-badge" style="border-color:rgba(255,182,39,0.3);'
            'color:#ffb627;background:rgba(255,182,39,0.06);">◌ Not Initialized</div>',
            unsafe_allow_html=True,
        )

    # ── PDF Upload & Indexing ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Vector Database</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs to index",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded_files:
        already    = {d["name"] for d in st.session_state.loaded_docs}
        new_files  = [f for f in uploaded_files if f.name not in already]

        if new_files:
            # Create vectorstore lazily if not done yet
            if st.session_state.vectorstore is None:
                with st.spinner("Initializing vector store..."):
                    from langchain_community.vectorstores import Chroma
                    st.session_state.vectorstore = Chroma(
                        persist_directory="chroma_db",
                        embedding_function=get_embeddings(),
                    )

            for f in new_files:
                with st.spinner(f"Indexing {f.name}…"):
                    try:
                        n = index_pdf(f, st.session_state.vectorstore)
                        st.session_state.loaded_docs.append({"name": f.name, "chunks": n})
                        st.success(f"✅ {f.name} — {n} chunks added")

                        # Rebuild retriever immediately so new docs are searchable
                        if st.session_state.rag_ready:
                            st.session_state.retriever = build_retriever(
                                st.session_state.vectorstore,
                                st.session_state.search_type_val,
                                st.session_state.k_val,
                                st.session_state.fetch_k_val,
                                st.session_state.lambda_val,
                            )
                    except Exception as e:
                        st.error(f"Failed to index {f.name}: {e}")

    # Loaded docs list
    if st.session_state.loaded_docs:
        for doc in st.session_state.loaded_docs:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:7px 10px;
                        background:#131920;border:1px solid #1e2d3d;border-radius:6px;margin:4px 0;">
                <div style="background:rgba(255,77,109,0.15);color:#ff4d6d;font-size:9px;
                            font-weight:700;padding:2px 5px;border-radius:4px;flex-shrink:0;">PDF</div>
                <div style="flex:1;font-size:11px;color:#c9d8e8;overflow:hidden;
                            text-overflow:ellipsis;white-space:nowrap;"
                     title="{doc['name']}">{doc['name']}</div>
                <div style="font-size:9px;color:#364d62;flex-shrink:0;">{doc['chunks']}c</div>
                <div style="width:6px;height:6px;border-radius:50%;background:#00e5a0;flex-shrink:0;"></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        chroma_exists = os.path.exists("chroma_db")
        label  = "chroma_db/ found — click ⚡ Init RAG" if chroma_exists else "No chroma_db — upload a PDF first"
        colour = "#00e5a0" if chroma_exists else "#ffb627"
        dot    = "●" if chroma_exists else "○"
        st.markdown(f"""
        <div style="padding:7px 10px;background:#131920;border:1px solid #1e2d3d;
                    border-radius:6px;font-size:11px;color:#364d62;
                    display:flex;align-items:center;gap:6px;">
            <span style="color:{colour};">{dot}</span>{label}
        </div>
        """, unsafe_allow_html=True)

    # ── Retrieval Config ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Retrieval Config</div>', unsafe_allow_html=True)

    search_type = st.selectbox(
        "Search Type",
        options=["mmr", "similarity", "similarity_score_threshold"],
        format_func=lambda x: {
            "mmr":                        "MMR — Max Marginal Relevance",
            "similarity":                 "Similarity Search",
            "similarity_score_threshold": "Similarity + Score Threshold",
        }[x],
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        k       = st.slider("k (results)", 1, 10, 4)
    with col2:
        fetch_k = st.slider("fetch_k", 4, 30, 10)

    lambda_mult = st.slider("λ diversity (MMR)", 0.0, 1.0, 0.5, 0.05)

    # Persist config so the PDF indexer rebuild can reference them
    st.session_state.search_type_val = search_type
    st.session_state.k_val           = k
    st.session_state.fetch_k_val     = fetch_k
    st.session_state.lambda_val      = lambda_mult

    # ── Session Stats ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Session Stats</div>', unsafe_allow_html=True)

    avg_lat  = (round(st.session_state.total_latency / st.session_state.query_count)
                if st.session_state.query_count else 0)
    hit_rate = (round((st.session_state.hit_count / st.session_state.query_count) * 100)
                if st.session_state.query_count else 100)

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-val">{st.session_state.query_count}</div>
            <div class="stat-label">Queries</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{k}</div>
            <div class="stat-label">Chunks/Q</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{avg_lat}ms</div>
            <div class="stat-label">Avg Latency</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{hit_rate}%</div>
            <div class="stat-label">Hit Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Init / Clear buttons ───────────────────────────────────────────────────
    col_init, col_clear = st.columns(2)

    with col_init:
        if st.button("⚡ Init RAG", use_container_width=True):
            with st.spinner("Loading pipeline..."):
                try:
                    from langchain_community.vectorstores import Chroma

                    # Reuse existing vectorstore (keeps any newly uploaded PDFs)
                    if st.session_state.vectorstore is None:
                        vs = Chroma(
                            persist_directory="chroma_db",
                            embedding_function=get_embeddings(),
                        )
                        st.session_state.vectorstore = vs
                    else:
                        vs = st.session_state.vectorstore

                    st.session_state.retriever = build_retriever(vs, search_type, k, fetch_k, lambda_mult)
                    st.session_state.llm       = get_llm()
                    st.session_state.prompt    = get_prompt()
                    st.session_state.rag_ready = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Init failed: {e}")

    with col_clear:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages      = []
            st.session_state.query_count   = 0
            st.session_state.total_latency = 0
            st.session_state.hit_count     = 0
            st.rerun()

    st.markdown("""
    <div style="font-size:0.62rem;color:#364d62;margin-top:12px;line-height:1.8;">
        Chroma DB · Persist: ./chroma_db<br>
        Splitter: Recursive · chunk=1000 · overlap=200<br>
        Model: mistral-small-2506
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="rag-title" style="font-size:1.8rem;">Ask your <span>documents</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="rag-subtitle">Chroma vector DB · MistralAI embeddings · MMR retrieval</div>',
    unsafe_allow_html=True,
)

# ── Empty / welcome state ──────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:40px 20px;color:#364d62;font-size:0.8rem;line-height:2.2;">
        📄 &nbsp; Upload a PDF <strong>or</strong> use existing chroma_db
        &nbsp;·&nbsp; Click <strong style="color:#00d4ff;">⚡ Init RAG</strong>
        &nbsp;·&nbsp; Then ask anything
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:0.65rem;color:#364d62;text-align:center;'
        'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Try asking</div>',
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

        # Build source chips HTML
        source_html = ""
        if msg.get("sources"):
            chips_html = "".join(
                f'<span class="source-chip">● {s}</span>'
                for s in msg["sources"]
            )
            source_html = (
                '<div class="sources-block">'
                '<div class="sources-label">Retrieved Sources</div>'
                f'<div class="source-chips">{chips_html}</div>'
                '</div>'
            )

        not_found_html = (
            '<div class="not-found">⚠ No relevant content found in the loaded documents.</div>'
            if is_not_found else ""
        )

        latency_ms  = msg.get("latency", 0)
        latency_str = (
            f' &nbsp;·&nbsp; <span style="color:#00d4ff;">{latency_ms}ms</span>'
            if latency_ms else ""
        )

        st.markdown(f"""
        <div class="msg-assistant">
            <div class="msg-header">AI · RAGIntel{latency_str}</div>
            <div>{safe_html(msg["content"])}</div>
            {not_found_html}
            {source_html}
        </div>
        """, unsafe_allow_html=True)

        # Retrieved chunks expander
        if msg.get("chunks"):
            with st.expander(f"📦 Retrieved chunks ({len(msg['chunks'])})"):
                for _i, _chunk in enumerate(msg["chunks"]):
                    st.markdown(f"**Chunk #{_i + 1}**")
                    st.markdown(
                        f'<div class="chunk-box">'
                        f'{safe_html(_chunk[:500])}'
                        f'{"…" if len(_chunk) > 500 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT
# ══════════════════════════════════════════════════════════════════════════════
prefill = st.session_state.pop("prefill", "")
query   = st.chat_input("Ask a question about your documents…") or prefill

if query:
    if not st.session_state.rag_ready:
        st.warning("⚡ Please click **Init RAG** in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Retrieving context & generating answer…"):
            t0 = time.time()
            try:
                docs    = st.session_state.retriever.invoke(query)
                context = "\n\n".join([d.page_content for d in docs])
                fmt     = st.session_state.prompt.invoke({"context": context, "question": query})
                result  = st.session_state.llm.invoke(fmt)

                latency = round((time.time() - t0) * 1000)
                answer  = result.content
                sources = list({
                    d.metadata.get("source", "document")
                     .replace("\\", "/").split("/")[-1]
                    for d in docs
                })
                chunks  = [d.page_content for d in docs]

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources,
                    "chunks":  chunks,
                    "latency": latency,
                })
                st.session_state.query_count   += 1
                st.session_state.total_latency += latency
                if "could not find" not in answer.lower():
                    st.session_state.hit_count += 1

            except Exception as e:
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": f"⚠ Error: {e}",
                    "sources": [],
                    "chunks":  [],
                    "latency": 0,
                })

        st.rerun()
