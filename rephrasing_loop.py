"""
Rephrasing Loop module for VAID JI Medical Research Assistant
Handles query refinement and quality scoring for better responses
"""

import os
import re
import time
import logging
from typing import List

from rag_pipeline import run_rag_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
try:
    from config import MAX_REPHRASE_ATTEMPTS, MIN_RESPONSE_LENGTH, MEDICAL_KEYWORDS
except ImportError:
    logger.warning("config.py not found, using default values")
    MAX_REPHRASE_ATTEMPTS = 2
    MIN_RESPONSE_LENGTH = 50
    MEDICAL_KEYWORDS = ['study', 'treatment', 'diagnosis', 'clinical', 'dosage', 'symptom', 'trial']

# === REPHRASE TRIGGERS ===
REPHRASE_TRIGGERS = [
    r"\bwhat is this\??",
    r"\bcan you explain\??",
    r"\btell me more\??",
    r"\bwhat do you mean\??",
    r"\bwhy\??"
]


def is_response_low_quality(response: str) -> bool:
    """
    Score response based on length and medical term presence.
    
    Args:
        response: The response text to evaluate
        
    Returns:
        bool: True if response is low quality, False otherwise
    """
    response = response.lower()
    length_ok = len(response.strip()) >= MIN_RESPONSE_LENGTH
    contains_medical_term = any(keyword in response for keyword in MEDICAL_KEYWORDS)
    return not (length_ok and contains_medical_term)


def rephrase_query(query: str, attempt: int) -> str:
    """
    Rephrase query to improve response quality.
    
    Args:
        query: The original query string
        attempt: The current attempt number
        
    Returns:
        str: The rephrased query
    """
    query = query.strip().lower()

    for pattern in REPHRASE_TRIGGERS:
        if re.search(pattern, query):
            return f"Please provide more context or specify details regarding the medical topic in question. (Attempt {attempt})"

    # If not vague, attempt to enrich it
    return f"{query} (Please elaborate using medical terminology.)"


def smart_medical_query(query: str) -> str:
    """
    Handle query with rephrasing loop and quality scoring.
    
    Args:
        query: The user's medical question
        
    Returns:
        str: The final response after quality checks and possible rephrasing
    """
    original_query = query
    response = ""
    attempt = 0

    while attempt <= MAX_REPHRASE_ATTEMPTS:
        try:
            logger.info(f"Attempt {attempt + 1}: Querying RAG pipeline...")
            response = run_rag_pipeline(query)
        except Exception as e:
            logger.error(f"Failed to run RAG pipeline: {str(e)}")
            return "[System Error] Unable to process the request."

        logger.info(f"Response received: {response[:100]}...")

        if not is_response_low_quality(response):
            return response

        logger.warning("Low quality response detected.")
        attempt += 1
        if attempt > MAX_REPHRASE_ATTEMPTS:
            break

        query = rephrase_query(original_query, attempt)
        logger.info(f"Rephrased Query: {query}")

    return "[Final Response] Unable to generate a high-quality medical answer. Please provide more context."


# === ENTRY POINT ===
if __name__ == "__main__":
    print("=== Smart Medical Query (with Rephrasing) ===")
    while True:
        user_input = input("\nAsk a medical question (or type 'exit'): ").strip()
        if user_input.lower() in {'exit', 'quit'}:
            break

        result = smart_medical_query(user_input)
        print("\n[Answer]:", result)
