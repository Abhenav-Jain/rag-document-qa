"""
create_database.py
──────────────────
Run this ONCE to build your Chroma vector database from a PDF.
After this, use app.py (Streamlit UI) for all queries.

Usage:
    python create_database.py
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
PDF_PATH       = "document loaders/book.pdf"   # change to your PDF path
CHROMA_DIR     = "chroma_db"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
EMBEDDING_MODEL = "mistral-embed"

# ── Load PDF ───────────────────────────────────────────────────────────────────
print(f"[1/4] Loading PDF: {PDF_PATH}")
loader = PyMuPDFLoader(PDF_PATH)
docs   = loader.load()
print(f"      Loaded {len(docs)} pages")

# ── Split into chunks ──────────────────────────────────────────────────────────
print(f"[2/4] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"      Created {len(chunks)} chunks")

# ── Create embeddings ──────────────────────────────────────────────────────────
print(f"[3/4] Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = MistralAIEmbeddings(model=EMBEDDING_MODEL)

# ── Store in Chroma ────────────────────────────────────────────────────────────
print(f"[4/4] Storing in Chroma DB at: ./{CHROMA_DIR}")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR,
)

print(f"\n✅ Done! Vector database saved to ./{CHROMA_DIR}")
print(f"   Now run:  streamlit run app.py")
