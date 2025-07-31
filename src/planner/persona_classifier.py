# FILE: src/planner/persona_classifier.py
# V1.1 (Resilience Fix): Added request_options to the generate_content call.

import logging
from typing import Literal
# --- DEFINITIVE FIX: Import the config and model getter ---
from src.tools.clients import get_flash_model, DEFAULT_REQUEST_OPTIONS

logger = logging.getLogger(__name__)

Persona = Literal["clinical_analyst", "health_economist", "regulatory_specialist"]

PERSONA_CLASSIFICATION_PROMPT = """
You are an expert request router. Your task is to analyze the user's question and determine which specialist persona is best equipped to answer it. You must choose from the available personas and provide ONLY the persona's key name as your response.

**Available Personas & Their Expertise:**

1.  **`clinical_analyst`**:
    *   Focuses on: Clinical trial data, drug efficacy, safety profiles, patient outcomes, medical conditions, and mechanisms of action.
    *   Keywords: treat, condition, indication, dosage, patients, trial, effective, side effects.
    *   Choose this persona for questions about the medical and scientific aspects of a drug.

2.  **`health_economist`**:
    *   Focuses on: Cost-effectiveness, pricing, market access, economic evaluations, and healthcare policy implications.
    *   Keywords: cost, price, economic, budget, financial, value, policy, summary.
    *   Choose this persona for questions about the financial or policy-level impact of a drug.

3.  **`regulatory_specialist`**:
    *   Focuses on: Submission types, meeting agendas, regulatory pathways (e.g., PBS listing types), sponsors, and official guidelines.
    *   Keywords: sponsor, submission, listing, agenda, meeting, guideline, change, status.
    *   Choose this persona for questions about the process and logistics of drug approval and listing.

**User Question:**
"{question}"

**Instructions:**
- Read the user's question carefully.
- Compare it against the expertise of each persona.
- Return ONLY the single key name (e.g., `clinical_analyst`) of the best-fitting persona. Do not add any explanation or other text.
"""

class PersonaClassifier:
    def __init__(self):
        # --- DEFINITIVE FIX: Use the new centralized model getter ---
        self.llm = get_flash_model()
        if not self.llm:
            logger.error("FATAL: Gemini client not initialized, PersonaClassifier will not work.")

    def classify(self, query: str) -> Persona:
        """Classifies the query and returns the most appropriate persona key."""
        if not self.llm:
            return "regulatory_specialist" # A safe default

        try:
            prompt = PERSONA_CLASSIFICATION_PROMPT.format(question=query)
            # --- DEFINITIVE FIX: Add request_options to the call ---
            response = self.llm.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            persona_key = response.text.strip()

            if persona_key in ["clinical_analyst", "health_economist", "regulatory_specialist"]:
                logger.info(f"Query classified for persona: '{persona_key}'")
                return persona_key
            else:
                logger.warning(f"Persona classification returned an invalid key: '{persona_key}'. Falling back to default.")
                return "regulatory_specialist"
        except Exception as e:
            logger.error(f"Persona classification failed: {e}", exc_info=True)
            return "regulatory_specialist" # Fallback on error