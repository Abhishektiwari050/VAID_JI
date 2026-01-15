"""
RAG (Retrieval-Augmented Generation) pipeline for VAID JI
Handles question answering with context retrieval from ChromaDB
"""

import os
import time
import hashlib
import pickle
import logging
from typing import Optional, List

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.openai import OpenAI as OpenRouterLLM
from langchain_community.llms.groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIG ===
try:
    from config import (
        CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, TEMPERATURE,
        CACHE_PATH, OPENROUTER_API_KEY, GROQ_API_KEY,
        RAG_PROMPT_TEMPLATE
    )
    PROMPT_TEMPLATE = RAG_PROMPT_TEMPLATE
except ImportError:
    # Fallback configuration
    CHROMA_DB_PATH = "./chroma_db"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    CACHE_PATH = "./rag_cache.pkl"
    TEMPERATURE = 0.1
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PROMPT_TEMPLATE = """You are a medical research assistant. Answer based ONLY on provided context. If context is missing, say so. Be precise and use medical terminology appropriately.

Context:
{context}

Question: {question}
"""

# === Cache for repeated questions ===
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        QUERY_CACHE = pickle.load(f)
else:
    QUERY_CACHE = {}


def get_cache_key(query: str) -> str:
    """
    Generate a SHA256 hash key for caching query results
    
    Args:
        query: The query string to hash
        
    Returns:
        str: Hexadecimal hash string
    """
    return hashlib.sha256(query.encode()).hexdigest()


def save_cache():
    """Save the query cache to disk"""
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(QUERY_CACHE, f)
        logger.debug("Cache saved successfully")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


# === LLM Wrappers ===

class MistralLLM(LLM):
    """Primary LLM: Mistral via OpenRouter"""
    
    @property
    def _llm_type(self) -> str:
        return "mistral"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call the Mistral LLM via OpenRouter API"""
        try:
            llm = OpenRouterLLM(
                model="mistralai/mistral-7b-instruct",
                temperature=TEMPERATURE,
                openai_api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            # Note: stop parameter not directly supported by OpenRouterLLM wrapper
            return llm.predict(prompt)
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            raise RuntimeError(f"Mistral API error: {str(e)}")


class LLaMAFallbackLLM(LLM):
    """Fallback LLM: LLaMA3 via Groq"""
    
    @property
    def _llm_type(self) -> str:
        return "llama3"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call the LLaMA3 LLM via Groq API"""
        try:
            llm = Groq(
                temperature=TEMPERATURE,
                model="llama3-8b-8192",
                groq_api_key=GROQ_API_KEY
            )
            # Note: stop parameter not directly supported by Groq wrapper
            return llm.predict(prompt)
        except Exception as e:
            logger.error(f"Groq fallback failed: {e}")
            raise RuntimeError(f"Groq fallback failed: {str(e)}")


def load_vectorstore():
    """Load ChromaDB vector store with embeddings"""
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
        return vectordb
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise


def run_rag_pipeline(query: str) -> str:
    """
    Run the RAG pipeline to answer a query using retrieved context
    
    Args:
        query: The user's question
        
    Returns:
        str: The generated answer
        
    Raises:
        RuntimeError: If both LLMs fail
    """
    # Check cache first
    key = get_cache_key(query)
    if key in QUERY_CACHE:
        logger.info("Returning cached result")
        return QUERY_CACHE[key]

    # Load vector store and create retriever
    try:
        vectordb = load_vectorstore()
        retriever = vectordb.as_retriever()
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return f"[Error] Failed to access knowledge base: {str(e)}"

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    # Try primary LLM (Mistral)
    try:
        llm = MistralLLM()
        logger.info("Using Mistral-7B-Instruct via OpenRouter...")
    except Exception as e:
        logger.warning(f"Mistral failed. Retrying with fallback in 1 second... ({e})")
        time.sleep(1)
        try:
            llm = LLaMAFallbackLLM()
            logger.info("Using LLaMA3-8B-8192 via Groq fallback...")
        except Exception as e2:
            logger.error(f"Both LLMs failed: Primary={e}, Fallback={e2}")
            return "[Error] All LLM services are unavailable. Please check your API keys."

    # Create QA chain and run
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )
        
        result = qa_chain.run(query)
        
        # Cache the result
        QUERY_CACHE[key] = result
        save_cache()
        
        return result
    except Exception as e:
        logger.error(f"QA chain execution failed: {e}")
        return f"[Error] Failed to generate answer: {str(e)}"


if __name__ == "__main__":
    print("=== RAG Medical Assistant ===")
    while True:
        question = input("\nAsk a medical question (or 'exit'): ")
        if question.strip().lower() in {"exit", "quit"}:
            break
        try:
            response = run_rag_pipeline(question)
            print("\n[Answer]:", response)
        except Exception as e:
            print(f"[ERROR] {e}")
