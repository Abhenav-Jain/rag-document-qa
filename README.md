# 📄 RAG Document Q&A

A **Retrieval-Augmented Generation (RAG)** system built while learning LangChain, Mistral AI, and vector databases. Ask questions from your PDF documents and get context-aware answers.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Mistral AI](https://img.shields.io/badge/Mistral_AI-FF7000?style=flat&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-00D4FF?style=flat&logoColor=white)

---

## 🛠️ Tech Stack

- **LLM** — Mistral AI (`mistral-small-2506`)
- **Embeddings** — Mistral AI (`mistral-embed`)
- **Vector Store** — ChromaDB
- **Framework** — LangChain
- **UI** — Streamlit

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/Abhenav-Jain/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 3. Build vector database

```bash
python create_database.py
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📁 Structure

```
├── app.py                  # Streamlit UI
├── create_database.py      # Build Chroma DB from PDF
├── main.py                 # CLI version
├── retrievers/             # Retrieval experiments
├── .env.example
└── requirements.txt
```
