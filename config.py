"""
Configuration module for VAID JI Medical Research Assistant
Centralizes all configuration settings and environment variables
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CACHE_PATH = "./rag_cache.pkl"
SENTENCE_TRANSFORMERS_CACHE = "./.sentence_transformers_cache"

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
MISTRAL_MODEL = "mistralai/mistral-7b-instruct"
LLAMA_MODEL = "llama3-8b-8192"

# LLM Settings
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

# Text Processing
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_TEXT_LENGTH = 100
SUMMARY_WORD_LIMIT = 200

# Query Settings
MAX_REPHRASE_ATTEMPTS = 2
MIN_RESPONSE_LENGTH = 50
RETRIEVAL_TOP_K = 4

# OCR Settings
OCR_RESOLUTION = 300
OCR_LANGUAGE = "eng"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Google Sheets (for testing automation)
GSHEET_CREDENTIALS = "credentials.json"
GSHEET_NAME = os.getenv("GSHEET_NAME", "PDF_Test_Results")

# Prompt Templates
RAG_PROMPT_TEMPLATE = """You are a medical research assistant. Answer based ONLY on provided context. If context is missing, say so. Be precise and use medical terminology appropriately.

Context:
{context}

Question: {question}
"""

SUMMARY_PROMPT_TEMPLATE = """Summarize key findings in plain language based ONLY on provided context.
Keep the summary brief and under 200 words.

Context:
{context}

Summary:
"""

# Medical Keywords for Quality Scoring
MEDICAL_KEYWORDS = [
    'study', 'treatment', 'diagnosis', 'clinical', 'dosage', 
    'symptom', 'trial', 'patient', 'therapy', 'medication'
]

def validate_config() -> tuple[bool, list[str]]:
    """
    Validate configuration and return status with any error messages
    
    Returns:
        tuple: (is_valid, list of error messages)
    """
    errors = []
    
    # Check API keys
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY not set in environment")
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY not set in environment (fallback LLM will not work)")
    
    # Check paths
    if not os.path.exists(BASE_DIR):
        errors.append(f"Base directory not found: {BASE_DIR}")
    
    # Check numeric values
    if TEMPERATURE < 0 or TEMPERATURE > 2:
        errors.append(f"Invalid TEMPERATURE value: {TEMPERATURE} (should be 0-2)")
    
    if CHUNK_SIZE < 100:
        errors.append(f"CHUNK_SIZE too small: {CHUNK_SIZE}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_config_summary() -> dict:
    """
    Get a summary of current configuration (safe to display)
    
    Returns:
        dict: Configuration summary without sensitive data
    """
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_primary": MISTRAL_MODEL,
        "llm_fallback": LLAMA_MODEL,
        "temperature": TEMPERATURE,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chroma_db_path": CHROMA_DB_PATH,
        "log_level": LOG_LEVEL,
        "api_keys_configured": {
            "openrouter": bool(OPENROUTER_API_KEY),
            "groq": bool(GROQ_API_KEY)
        }
    }


if __name__ == "__main__":
    """Validate configuration when run directly"""
    print("=" * 50)
    print("VAID JI Configuration Validation")
    print("=" * 50)
    
    is_valid, errors = validate_config()
    
    if is_valid:
        print("\n✅ Configuration is valid!")
        print("\nConfiguration Summary:")
        for key, value in get_config_summary().items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease check your .env file and ensure all required variables are set.")
