# FILE: src/planner/query_classifier.py
# Phase 1.1: QueryClassifier — interprets user query using Gemini and returns structured metadata

import logging
import google.generativeai as genai
from typing import Optional
from src.models import QueryMetadata
from src.prompts import QUERY_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)

class QueryClassifier:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def classify(self, query: str) -> Optional[QueryMetadata]:
        """Classifies a user query into intent, keywords, and graph suitability."""
        logger.info(f"Classifying query: {query}")
        try:
            prompt = QUERY_CLASSIFICATION_PROMPT + f"\n\nUser Query: {query}"
            response = self.model.generate_content(prompt)
            parsed_json = response.text.strip()
            metadata = QueryMetadata.model_validate_json(parsed_json)
            logger.info(f"Classification result: {metadata.model_dump_json(indent=2)}")
            return metadata
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return None

# Unit test: test with canned queries
if __name__ == "__main__":
    qc = QueryClassifier()
    test_queries = [
        "What company sponsors Abaloparatide?",
        "Compare the clinical outcomes of Drug A vs Drug B",
        "Tell me about submissions for lung cancer.",
        "What is the patient population for the March 2025 submission?"
    ]
    for q in test_queries:
        print(qc.classify(q))
