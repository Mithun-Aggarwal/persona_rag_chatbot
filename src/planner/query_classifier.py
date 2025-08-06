# FILE: src/planner/query_classifier.py
# V2.0 (Resilient Architecture): Refactored to use a centralized, resilient LLM
# calling mechanism. Can now gracefully handle transient API errors (e.g., 503
# Overloaded) by returning None, allowing the agent to terminate the run cleanly.

import logging
import json
import re
from typing import Optional

from google.api_core import exceptions as google_exceptions
from src.models import QueryMetadata
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

    # --- START OF DEFINITIVE FIX: Resilient LLM Call Helper ---
    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """A wrapper for generate_content that handles API retries and timeouts gracefully."""
        if not self.model:
            logger.error("LLM model for QueryClassifier is not available.")
            return None
        try:
            response = self.model.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            return response.text
        except google_exceptions.RetryError as e:
            logger.error(f"QueryClassifier API call timed out after multiple retries: {e}")
            return None # Signal recoverable failure
        except Exception as e:
            logger.error(f"An unexpected error occurred during QueryClassifier LLM call: {e}", exc_info=True)
            return None
    # --- END OF DEFINITIVE FIX ---

    def classify(self, query: str) -> Optional[QueryMetadata]:
        if not self.model: return None
        logger.info(f"Classifying query: {query}")
        
        prompt = QUERY_CLASSIFICATION_PROMPT + f"\n\nUser Query: {query}"
        
        # --- START OF DEFINITIVE FIX: Use the resilient wrapper ---
        response_text = self._call_llm_with_retry(prompt)
        
        if response_text is None:
            logger.error("Query classification failed due to API issues.")
            return None # Propagate the failure signal
        # --- END OF DEFINITIVE FIX ---

        try:
            json_data = extract_json_from_response(response_text)
            
            if "intent" in json_data and json_data["intent"] == "comparison":
                json_data["intent"] = "comparative_analysis"
            
            metadata = QueryMetadata.model_validate(json_data)
            
            logger.info(f"Classification result: {metadata.model_dump_json(indent=2)}")
            return metadata
        except Exception as e:
            # This catches Pydantic validation errors or other unexpected issues
            logger.error(f"Query classification failed on response data: {e}", exc_info=True)
            return None