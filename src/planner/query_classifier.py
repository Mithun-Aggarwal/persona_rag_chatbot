# FILE: src/planner/query_classifier.py
# V1.4 (Final Fix): Corrected Pydantic model import and literal values.

import logging
import json
import re
from typing import Optional

# --- DEFINITIVE FIX: Import the correct model from src.models ---
from src.models import QueryMetadata, QueryIntent
from src.prompts import QUERY_CLASSIFICATION_PROMPT_V2 as QUERY_CLASSIFICATION_PROMPT
from src.tools.clients import get_flash_model, DEFAULT_REQUEST_OPTIONS

logger = logging.getLogger(__name__)

def extract_json_from_response(text: str) -> dict:
    """Finds and parses the first valid JSON object from a string."""
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON from response text: {text}")
        return {}


class QueryClassifier:
    def __init__(self):
        self.model = get_flash_model()

    def classify(self, query: str) -> Optional[QueryMetadata]:
        if not self.model: return None
        logger.info(f"Classifying query: {query}")
        try:
            prompt = QUERY_CLASSIFICATION_PROMPT + f"\n\nUser Query: {query}"
            response = self.model.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            
            json_data = extract_json_from_response(response.text)
            
            # --- DEFINITIVE FIX: Ensure intent matches the allowed literals ---
            if "intent" in json_data and json_data["intent"] == "comparison":
                json_data["intent"] = "comparative_analysis"
            
            metadata = QueryMetadata.model_validate(json_data)
            
            logger.info(f"Classification result: {metadata.model_dump_json(indent=2)}")
            return metadata
        except Exception as e:
            logger.error(f"Query classification failed: {e}", exc_info=True)
            return None