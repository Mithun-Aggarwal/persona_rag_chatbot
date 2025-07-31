# FILE: src/planner/query_classifier.py
# V1.1 (Resilience Fix): Added request_options to the generate_content call.

import logging
from typing import Optional
from src.models import QueryMetadata
from src.prompts import QUERY_CLASSIFICATION_PROMPT
# --- DEFINITIVE FIX: Import the config and model getter ---
from src.tools.clients import get_flash_model, DEFAULT_REQUEST_OPTIONS

logger = logging.getLogger(__name__)

class QueryClassifier:
    def __init__(self):
        # --- DEFINITIVE FIX: Use the new centralized model getter ---
        self.model = get_flash_model()

    def classify(self, query: str) -> Optional[QueryMetadata]:
        """Classifies a user query into intent, keywords, and graph suitability."""
        if not self.model:
            logger.error("QueryClassifier model is not available.")
            return None
        logger.info(f"Classifying query: {query}")
        try:
            prompt = QUERY_CLASSIFICATION_PROMPT + f"\n\nUser Query: {query}"
            # --- DEFINITIVE FIX: Add request_options to the call ---
            response = self.model.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            parsed_json = response.text.strip()
            metadata = QueryMetadata.model_validate_json(parsed_json)
            logger.info(f"Classification result: {metadata.model_dump_json(indent=2)}")
            return metadata
        except Exception as e:
            logger.error(f"Query classification failed: {e}", exc_info=True)
            return None