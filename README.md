# 🧠 VAID JI: AI Medical Research Assistant

**VAID JI** is an open-source, zero-cost, RAG-based (Retrieval-Augmented Generation) assistant designed to help medical students, researchers, and the general public read, analyze, and understand medical PDFs — all without needing a GPU or paid API access.

---

## 🚀 Project Overview

**VAID JI** (वेद जी) acts as your intelligent, always-on medical assistant. It reads medical research papers and clinical PDFs, provides answers to your questions using embedded context, and even generates easy-to-understand summaries.

⚙️ This project runs entirely on CPU, is self-hostable, and built with privacy in mind — no data leaves your device.

---

## ✨ Features

- 📁 **Multi-PDF Upload** – Drag and drop multiple research PDFs for instant processing
- 🔍 **RAG-based Question Answering** – Ask questions and get context-aware answers
- 🧾 **Auto-Summarization** – Generate <200-word plain-English summaries of medical findings
- 🧠 **OCR Support** – Read and embed scanned or image-based PDFs
- 📱 **Responsive Mobile UI** – Access from desktop or phone
- 🧾 **PDF Export** – Export chat or summaries as formatted PDFs
- 💾 **Persistent ChromaDB** – Your embeddings are cached locally for speed and privacy

---

## 🧰 Tech Stack

| Layer | Tech |
|-------|------|
| **Language** | Python 3.10 |
| **Frontend** | Streamlit |
| **RAG Framework** | LangChain |
| **Vector Store** | ChromaDB (`chromadb==0.4.22`) |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **LLMs** | Mistral-7B-Instruct via OpenRouter (primary), LLaMA3-8B-8192 via Groq (fallback) |
| **OCR** | `pdfplumber`, `PyMuPDF` |
| **Utilities** | `pandas`, `PyPDF2`, `streamlit-extras`, `dotenv`, `pickle` |

---

## 🛠️ Setup Instructions

### 1. 📦 Clone the Repository

```bash
git clone https://github.com/Abhishektiwari050/VAID_JI.git
cd VAID_JI
