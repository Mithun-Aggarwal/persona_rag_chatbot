# FILE: src/planner/persona_classifier.py
# V2.1 (Final Production Grade): Completes the prompt centralization refactoring.
# This file now correctly imports the PERSONA_CLASSIFICATION_PROMPT from the
# central `src.prompts` module, making the system more maintainable and robust.

import logging
from typing import Literal, Optional

from google.api_core import exceptions as google_exceptions
from src.tools.clients import get_flash_model, DEFAULT_REQUEST_OPTIONS
# --- START OF DEFINITIVE FIX: Import from the central prompts module ---
from src.prompts import PERSONA_CLASSIFICATION_PROMPT
# --- END OF DEFINITIVE FIX ---


logger = logging.getLogger(__name__)

Persona = Literal["clinical_analyst", "health_economist", "regulatory_specialist"]


class PersonaClassifier:
    def __init__(self):
        self.llm = get_flash_model()
        if not self.llm:
            logger.error("FATAL: Gemini client not initialized, PersonaClassifier will not work.")

    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """A wrapper for generate_content that handles API retries and timeouts gracefully."""
        if not self.llm:
            logger.error("LLM model for PersonaClassifier is not available.")
            return None
        try:
            response = self.llm.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            return response.text
        except google_exceptions.RetryError as e:
            logger.error(f"PersonaClassifier API call timed out after multiple retries: {e}")
            return None # Return None to signal a recoverable failure
        except Exception as e:
            logger.error(f"An unexpected error occurred during PersonaClassifier LLM call: {e}", exc_info=True)
            return None

    def classify(self, query: str) -> Optional[Persona]:
        """
        Classifies the query and returns the most appropriate persona key.
        Returns None if the API call fails irrecoverably.
        """
        if not self.llm:
            return "regulatory_specialist"

        prompt = PERSONA_CLASSIFICATION_PROMPT.format(question=query)
        persona_key_text = self._call_llm_with_retry(prompt)
        
        if persona_key_text is None:
            # API call failed after retries, signal this to the agent
            return None 
            
        persona_key = persona_key_text.strip()

        if persona_key in ["clinical_analyst", "health_economist", "regulatory_specialist"]:
            logger.info(f"Query classified for persona: '{persona_key}'")
            return persona_key
        else:
            logger.warning(f"Persona classification returned an invalid key: '{persona_key}'. Falling back to default.")
            return "regulatory_specialist"