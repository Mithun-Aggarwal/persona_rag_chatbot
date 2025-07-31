# FILE: src/planner/query_classifier.py
# V1.2 (Metadata RAG): Added 'themes' to the output for filtered search.

import logging
from typing import Optional, List
from pydantic import BaseModel, Field

# --- DEFINITIVE FIX: Update the Pydantic model to include themes ---
class QueryMetadata(BaseModel):
    intent: str
    keywords: List[str]
    question_is_graph_suitable: bool
    themes: Optional[List[str]] = Field(default_factory=list, description="High-level themes like 'oncology', 'safety', 'pricing'.")

from src.prompts import QUERY_CLASSIFICATION_PROMPT_V2 as QUERY_CLASSIFICATION_PROMPT # Use new prompt version
from src.tools.clients import get_flash_model, DEFAULT_REQUEST_OPTIONS

logger = logging.getLogger(__name__)

class QueryClassifier:
    def __init__(self):
        self.model = get_flash_model()

    def classify(self, query: str) -> Optional[QueryMetadata]:
        if not self.model: return None
        logger.info(f"Classifying query: {query}")
        try:
            prompt = QUERY_CLASSIFICATION_PROMPT + f"\n\nUser Query: {query}"
            response = self.model.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            metadata = QueryMetadata.model_validate_json(response.text.strip())
            logger.info(f"Classification result: {metadata.model_dump_json(indent=2)}")
            return metadata
        except Exception as e:
            logger.error(f"Query classification failed: {e}", exc_info=True)
            return None