# rephrasing_loop.py

import os
import re
import time
from typing import List

from rag_pipeline import run_rag_pipeline  # Assumes rag_pipeline.py is in same folder

# === CONFIG ===
REPHRASE_TRIGGERS = [
    r"\bwhat is this\??",
    r"\bcan you explain\??",
    r"\btell me more\??",
    r"\bwhat do you mean\??",
    r"\bwhy\??"
]

MEDICAL_KEYWORDS = ['study', 'treatment', 'diagnosis', 'clinical', 'dosage', 'symptom', 'trial']
MIN_RESPONSE_LENGTH = 50
MAX_REPHRASE_ATTEMPTS = 2


# === QUALITY SCORING ===
def is_response_low_quality(response: str) -> bool:
    """Score response based on length and medical term presence."""
    response = response.lower()
    length_ok = len(response.strip()) >= MIN_RESPONSE_LENGTH
    contains_medical_term = any(keyword in response for keyword in MEDICAL_KEYWORDS)
    return not (length_ok and contains_medical_term)


# === REPHRASING STRATEGY ===
def rephrase_query(query: str, attempt: int) -> str:
    """Basic rephrasing logic."""
    query = query.strip().lower()

    for pattern in REPHRASE_TRIGGERS:
        if re.search(pattern, query):
            return f"Please provide more context or specify details regarding the medical topic in question. (Attempt {attempt})"

    # If not vague, attempt to enrich it
    return f"{query} (Please elaborate using medical terminology.)"


# === MAIN EXECUTION LOOP ===
def smart_medical_query(query: str) -> str:
    """Handles rephrasing loop + quality scoring."""

    original_query = query
    response = ""
    attempt = 0

    while attempt <= MAX_REPHRASE_ATTEMPTS:
        try:
            print(f"\n[Attempt {attempt + 1}] Querying RAG pipeline...")
            response = run_rag_pipeline(query)
        except Exception as e:
            print(f"[ERROR] Failed to run RAG pipeline: {str(e)}")
            return "[System Error] Unable to process the request."

        print(f"[Response]: {response}")

        if not is_response_low_quality(response):
            return response

        print("[Warning] Low quality response detected.")
        attempt += 1
        if attempt > MAX_REPHRASE_ATTEMPTS:
            break

        query = rephrase_query(original_query, attempt)
        print(f"[Rephrased Query]: {query}")

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
