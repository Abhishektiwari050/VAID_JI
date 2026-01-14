"""
Auto-summarization module for VAID JI Medical Research Assistant
Generates concise summaries of medical research documents
"""

import os
import traceback
import logging
from typing import List

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain_community.llms.openai import OpenAI as OpenRouterLLM
from langchain_community.llms.groq import Groq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from rag_pipeline import get_cache_key
except ImportError:
    # Fallback if rag_pipeline is not available
    import hashlib
    def get_cache_key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIG ===
try:
    from config import (
        CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, TEMPERATURE,
        CHUNK_SIZE, SUMMARY_WORD_LIMIT, OPENROUTER_API_KEY, 
        GROQ_API_KEY, SUMMARY_PROMPT_TEMPLATE
    )
    CHUNK_SIZE_LIMIT = CHUNK_SIZE
    PROMPT_TEMPLATE = SUMMARY_PROMPT_TEMPLATE
except ImportError:
    # Fallback to environment variables if config.py not available
    CHROMA_DB_PATH = "./chroma_db"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    TEMPERATURE = 0.1
    CHUNK_SIZE_LIMIT = 800
    SUMMARY_WORD_LIMIT = 200
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PROMPT_TEMPLATE = """Summarize key findings in plain language based ONLY on provided context.
Keep the summary brief and under 200 words.

Context:
{context}

Summary:
"""

# === LLM WRAPPERS ===

class MistralLLM(LLM):
    """Custom LLM wrapper for Mistral via OpenRouter"""
    
    @property
    def _llm_type(self) -> str:
        return "mistral"
    
    def _call(self, prompt: str, stop: List[str] | None = None, **kwargs) -> str:
        """Call the Mistral LLM via OpenRouter API"""
        try:
            llm = OpenRouterLLM(
                model="mistralai/mistral-7b-instruct",
                temperature=TEMPERATURE,
                openai_api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            return llm.predict(prompt)
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            raise RuntimeError(f"Mistral API error: {str(e)}")


class LLaMAFallbackLLM(LLM):
    """Custom LLM wrapper for LLaMA3 via Groq (fallback)"""
    
    @property
    def _llm_type(self) -> str:
        return "llama3"
    
    def _call(self, prompt: str, stop: List[str] | None = None, **kwargs) -> str:
        """Call the LLaMA3 LLM via Groq API"""
        try:
            llm = Groq(
                temperature=TEMPERATURE,
                model="llama3-8b-8192",
                groq_api_key=GROQ_API_KEY
            )
            return llm.predict(prompt)
        except Exception as e:
            logger.error(f"Groq fallback failed: {e}")
            raise RuntimeError(f"Groq fallback failed: {str(e)}")


def get_relevant_chunks(pdf_filename: str, max_chunks: int = 10) -> List[str]:
    """Retrieve top matching chunks from ChromaDB using filename as query."""
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
        retriever = vectordb.as_retriever()

        # Treat filename as query string for matching
        docs = retriever.get_relevant_documents(query=pdf_filename)
        chunks = [doc.page_content[:CHUNK_SIZE_LIMIT] for doc in docs[:max_chunks]]
        return chunks

    except Exception as e:
        raise RuntimeError(f"Chunk retrieval failed: {str(e)}")


def summarize_chunks(chunks: List[str]) -> str:
    """Run summarization using prompt + LLM with fallback."""
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context"])

    for attempt in range(2):
        try:
            llm = MistralLLM() if attempt == 0 else LLaMAFallbackLLM()
            print(f"Using {'Mistral-7B-Instruct' if attempt == 0 else 'LLaMA3-8B (fallback)'}...")
            chain = LLMChain(llm=llm, prompt=prompt)
            summaries = [chain.run(context=chunk) for chunk in chunks]
            return "\n".join(summaries)
        except Exception as e:
            print(f"[ERROR] LLM {attempt + 1} failed: {e}")
            if attempt == 1:
                raise

    return "[ERROR] All LLM attempts failed."


def generate_summary(pdf_filename: str) -> str:
    """High-level summary generator function."""
    try:
        print(f"\n📄 Summarizing: {pdf_filename}")
        chunks = get_relevant_chunks(pdf_filename)

        if not chunks:
            return "[No relevant context found. Ensure PDF embeddings are generated in ./chroma_db.]"

        raw_summary = summarize_chunks(chunks)

        # Trim summary to ~200 words
        words = raw_summary.split()
        if len(words) > SUMMARY_WORD_LIMIT:
            raw_summary = " ".join(words[:SUMMARY_WORD_LIMIT]) + "..."

        return raw_summary.strip()

    except Exception as e:
        traceback.print_exc()
        return f"[Summary Error] {str(e)}"


# === CLI ENTRY ===
if __name__ == "__main__":
    print("=== Medical PDF Summarizer ===")
    while True:
        filename = input("\nEnter PDF filename (or 'exit'): ").strip()
        if filename.lower() in {"exit", "quit"}:
            break

        summary = generate_summary(filename)
        print("\n[Summary]:\n", summary)
