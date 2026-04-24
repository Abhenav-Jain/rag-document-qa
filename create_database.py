"""
create_database.py
──────────────────
Builds Chroma vector DB from a PDF (run once).

Usage:
python create_database.py
"""

import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

PDF_PATH = r"D:\Coding\RAG Project\document loaders\book.pdf"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "mistral-embed"

# ── VALIDATION ────────────────────────────────────────────────────────────────

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"❌ PDF not found at: {PDF_PATH}")

# ── LOAD PDF ──────────────────────────────────────────────────────────────────

print(f"[1/5] Loading PDF: {PDF_PATH}")
loader = PyMuPDFLoader(PDF_PATH)
docs = loader.load()
print(f"      Loaded {len(docs)} pages")

# ── ADD METADATA ──────────────────────────────────────────────────────────────

for doc in docs:
    doc.metadata["source"] = os.path.basename(PDF_PATH)

# ── SPLIT INTO CHUNKS ─────────────────────────────────────────────────────────

print(f"[2/5] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

splitter = RecursiveCharacterTextSplitter(
chunk_size=CHUNK_SIZE,
chunk_overlap=CHUNK_OVERLAP,
separators=["\n\n", "\n", " ", ""],
)

chunks = splitter.split_documents(docs)
print(f"      Created {len(chunks)} chunks")

# ── LOAD EMBEDDINGS ───────────────────────────────────────────────────────────

print(f"[3/5] Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = MistralAIEmbeddings(model=EMBEDDING_MODEL)

# ── CLEAR OLD DB ──────────────────────────────────────────────────────────────

if os.path.exists(CHROMA_DIR):
    print(f"[4/5] Removing old database at ./{CHROMA_DIR}")
    shutil.rmtree(CHROMA_DIR)

# ── STORE IN CHROMA ───────────────────────────────────────────────────────────

print(f"[5/5] Storing in Chroma DB at: ./{CHROMA_DIR}")

vector_store = Chroma.from_documents(
documents=chunks,
embedding=embedding_model,
persist_directory=CHROMA_DIR,
)

# ── DONE ──────────────────────────────────────────────────────────────────────

print("\n✅ Done! Vector database created successfully.")
print(f"📁 Location: ./{CHROMA_DIR}")
print(f"📊 Total chunks: {len(chunks)}")

print("\n🚀 Next Step:")
print("👉 Run your app using: streamlit run app.py")
