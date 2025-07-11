# rag_pipeline.py

import os
import time
import hashlib
import pickle

from typing import Optional

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.llms import OpenAI  # Dummy base class for type hint

from langchain_community.llms.openai import OpenAI as OpenRouterLLM
from langchain_community.llms.groq import Groq

# Ensure chromadb and sentence-transformers are installed:
# pip install chromadb==0.4.22 sentence-transformers==2.2.2 langchain==0.1.14

# === CONFIG ===
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_PATH = "./rag_cache.pkl"
TEMPERATURE = 0.1

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Prompt Template ===
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
    return hashlib.sha256(query.encode()).hexdigest()


def save_cache():
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(QUERY_CACHE, f)


# === LLM Wrappers ===

class MistralLLM(LLM):
    """Primary LLM: Mistral via OpenRouter"""
    def _call(self, prompt: str, **kwargs) -> str:
        try:
            llm = OpenRouterLLM(
                model="mistralai/mistral-7b-instruct",
                temperature=TEMPERATURE,
                openai_api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            return llm.predict(prompt)
        except Exception as e:
            raise RuntimeError(f"Mistral API error: {str(e)}")


class LLaMAFallbackLLM(LLM):
    """Fallback LLM: LLaMA3 via Groq"""
    def _call(self, prompt: str, **kwargs) -> str:
        try:
            llm = Groq(
                temperature=TEMPERATURE,
                model="llama3-8b-8192",
                groq_api_key=GROQ_API_KEY
            )
            return llm.predict(prompt)
        except Exception as e:
            raise RuntimeError(f"Groq fallback failed: {str(e)}")


def load_vectorstore():
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    return vectordb


def run_rag_pipeline(query: str) -> str:
    key = get_cache_key(query)
    if key in QUERY_CACHE:
        return QUERY_CACHE[key]

    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever()

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    try:
        llm = MistralLLM()
        print("Using Mistral-7B-Instruct via OpenRouter...")
    except Exception as e:
        print(f"Mistral failed. Retrying with fallback in 1 second... ({e})")
        time.sleep(1)
        llm = LLaMAFallbackLLM()
        print("Using LLaMA3-8B-8192 via Groq fallback...")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.run(query)
    QUERY_CACHE[key] = result
    save_cache()
    return result


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
