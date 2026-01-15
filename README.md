# 🧠 VAID JI: AI Medical Research Assistant

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Stars](https://img.shields.io/github/stars/Abhishektiwari050/VAID_JI?style=social)
![Issues](https://img.shields.io/github/issues/Abhishektiwari050/VAID_JI)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

**VAID JI** (वेद जी) is an open-source, zero-cost, RAG-based (Retrieval-Augmented Generation) assistant designed to help medical students, researchers, and the general public read, analyze, and understand medical PDFs — all without needing a GPU or paid API access.

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

### Prerequisites

- Python 3.10 or higher
- 16GB RAM (recommended)
- Internet connection for initial setup and API access

### 1. 📦 Clone the Repository

```bash
git clone https://github.com/Abhishektiwari050/VAID_JI.git
cd VAID_JI
```

### 2. 🔧 Create Virtual Environment

```bash
# Create virtual environment
python -m venv medical_assistant_env

# Activate it
# On Windows:
medical_assistant_env\Scripts\activate
# On Linux/Mac:
source medical_assistant_env/bin/activate
```

### 3. 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. 🔑 Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys:
# - Get OpenRouter API key from: https://openrouter.ai/
# - Get Groq API key from: https://console.groq.com/
```

### 5. 🚀 Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Uploading PDFs

1. Navigate to the **Upload Documents** tab
2. Drag and drop PDF files or click to browse
3. Wait for processing to complete
4. View document metadata and preview

### Asking Questions

1. Go to the **Ask Questions** tab
2. Type your medical research question
3. Click **Get Answer** 
4. View AI-generated response with context

### Viewing History

1. Navigate to the **Query History** tab
2. Browse previous questions and answers
3. Export history if needed

---

## 🏗️ Project Structure

```
VAID_JI/
├── app.py                      # Main Streamlit application
├── pdf_preprocessing.py        # PDF text extraction and chunking
├── ocr_processing.py          # OCR for scanned PDFs
├── rag_pipeline.py            # RAG implementation with LLMs
├── auto_summary.py            # Automatic summarization
├── question_classifier.py     # Question type classification
├── rephrasing_loop.py         # Query refinement logic
├── pdf_test_automation.py     # Google Sheets integration for testing
├── phase1_setup.py            # Environment setup script
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
├── CONTRIBUTING.md           # Contribution guidelines
└── README.md                 # This file
```

---

## 🔧 Configuration

Edit `.env` file to customize:

- **API Keys**: OpenRouter and Groq API keys
- **Model Settings**: Temperature, max tokens
- **Database Path**: ChromaDB storage location
- **Embedding Model**: Default is `all-MiniLM-L6-v2`
- **Logging Level**: DEBUG, INFO, WARNING, ERROR

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `Tesseract not found`
- **Solution**: Install Tesseract OCR:
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr`
  - Mac: `brew install tesseract`

**Issue**: `API key not found`
- **Solution**: Ensure `.env` file exists with valid API keys

**Issue**: `ChromaDB persistence error`
- **Solution**: Check write permissions for `./chroma_db` directory

**Issue**: `Out of memory`
- **Solution**: Reduce batch size in configuration or close other applications

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://langchain.com/)
- Embeddings by [Sentence Transformers](https://www.sbert.net/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- LLMs via [OpenRouter](https://openrouter.ai/) and [Groq](https://groq.com/)

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Abhishektiwari050/VAID_JI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Abhishektiwari050/VAID_JI/discussions)
- **Author**: Abhishek Tiwari

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for the medical research community**
