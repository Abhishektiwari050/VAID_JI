# Quick Start Guide for VAID JI

This guide will help you get VAID JI up and running in minutes.

## Prerequisites

- Python 3.10 or higher
- 16GB RAM (recommended)
- Internet connection

## Installation Methods

### Option 1: Standard Installation (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abhishektiwari050/VAID_JI.git
   cd VAID_JI
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv medical_assistant_env
   
   # On Windows:
   medical_assistant_env\Scripts\activate
   
   # On Linux/Mac:
   source medical_assistant_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Option 2: Docker Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abhishektiwari050/VAID_JI.git
   cd VAID_JI
   ```

2. **Create .env file**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Open browser at http://localhost:8501

### Option 3: Automated Setup Script

1. **Clone and run setup**
   ```bash
   git clone https://github.com/Abhishektiwari050/VAID_JI.git
   cd VAID_JI
   python phase1_setup.py
   ```

2. **Activate environment and configure**
   ```bash
   # Activate as shown in setup output
   cp .env.example .env
   # Edit .env
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Getting API Keys

### OpenRouter API Key (Primary LLM)
1. Visit https://openrouter.ai/
2. Sign up for an account
3. Navigate to API Keys section
4. Create a new API key
5. Add to `.env` file as `OPENROUTER_API_KEY`

### Groq API Key (Fallback LLM)
1. Visit https://console.groq.com/
2. Sign up for an account
3. Go to API Keys
4. Generate a new API key
5. Add to `.env` file as `GROQ_API_KEY`

## First Steps

1. **Upload a PDF**
   - Go to "Upload Documents" tab
   - Drag and drop or select a medical PDF
   - Wait for processing

2. **Ask a Question**
   - Navigate to "Ask Questions" tab
   - Type your medical research question
   - Click "Get Answer"

3. **View History**
   - Check "Query History" tab for previous Q&A

## Troubleshooting

### Common Issues

**Port 8501 already in use**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

**API key errors**
- Verify keys are correctly set in `.env`
- Check for extra spaces or quotes
- Ensure `.env` is in the project root

**Tesseract not found (OCR)**
- Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

**Memory errors**
- Close other applications
- Reduce batch size in config
- Use smaller PDFs

## Next Steps

- Read the full [README.md](README.md) for detailed features
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Report issues on [GitHub Issues](https://github.com/Abhishektiwari050/VAID_JI/issues)

## Support

Need help? 
- Check the [Troubleshooting Guide](README.md#troubleshooting)
- Search [existing issues](https://github.com/Abhishektiwari050/VAID_JI/issues)
- Create a new issue with details

---

**Happy researching! 📚🔬**
