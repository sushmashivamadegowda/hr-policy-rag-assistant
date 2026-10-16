# 🏢 HR Policy RAG Assistant

A local Retrieval-Augmented Generation (RAG) web application built using **Streamlit**, **FAISS**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, and **LangChain** text splitters. It empowers employees and HR teams to query, search, and analyze HR Policy documents (PDF, DOCX, HTML, TXT) with precise passage citations and similarity scoring.

---

## ✨ Features

- 📑 **Multi-Format Document Support**: Upload PDF, DOCX, HTML, and TXT policy files.
- ⚡ **Preloaded Sample Policy**: Built-in comprehensive company HR policy (Leave, Remote Work, Expenses, Code of Conduct, Health Insurance) for instant testing.
- 🔍 **Vector Indexing (FAISS)**: Fast similarity search using 384-dimensional dense vectors from SentenceTransformers (`all-MiniLM-L6-v2`).
- 🤖 **Flexible RAG Generation Engine**:
  - **Structured Context Synthesizer**: Fast, accurate, factual extraction directly referencing section and page numbers.
  - **Local Flan-T5 Generator**: HuggingFace text-to-text model generation option.
- 💬 **Interactive Chat Interface**: Sleek Streamlit chat UI with quick-prompt chips, collapsible source passages, and similarity match percentages.
- 🎛️ **RAG Controls**: Dynamic sliders for chunk size, chunk overlap, top-K retrieval count, and similarity thresholding.
- 📊 **Vector Store Explorer**: Direct inspection and testing of FAISS chunk embeddings and similarity rankings.

---

## 🚀 Quick Start & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web App
```bash
streamlit run app.py
```

The app will launch locally at `http://localhost:8501`.

---

## 📁 Project Structure

```
d:/AIML/projects/RAG/
├── app.py                 # Streamlit Web Application UI & Chat logic
├── rag_engine.py          # Vector embedding, FAISS indexing, retrieval & synthesis engine
├── document_parsers.py    # Parsers for PDF, DOCX, HTML, and TXT documents
├── sample_policies.py    # Built-in sample HR Policy document
├── requirements.txt       # Project Python dependencies
└── README.md              # Project documentation
```

---

## 🛠️ Architecture & Workflow

```
[ Uploaded HR Doc (PDF/HTML/DOCX) ] 
               │
               ▼
   [ Text Splitter / Chunking ]  (Chunk Size: 400, Overlap: 50)
               │
               ▼
   [ SentenceTransformer ]       (all-MiniLM-L6-v2)
               │
               ▼
     [ FAISS Vector Index ]      (Flat IP / Cosine Similarity)
               │
      (User Query Input)
               │
               ▼
    [ Vector Search & Top-K ]    (Ranked by Cosine Score)
               │
               ▼
[ Context Synthesizer + Sources ] (Markdown Answer + Citations)
```
