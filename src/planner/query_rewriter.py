# FILE: src/planner/query_rewriter.py
# V2.0 (Performance Fix): Added a heuristic check to bypass the LLM call for non-conversational queries.

import logging
from typing import List, Optional
# --- DEFINITIVE FIX: Import the config and model getter ---
from src.tools.clients import get_flash_model, DEFAULT_REQUEST_OPTIONS

logger = logging.getLogger(__name__)

# --- DEFINITIVE FIX: Heuristic pre-flight check ---
# A list of common words that indicate a query is conversational and likely needs context from chat history.
# We check for these words (as whole words, case-insensitively) before making a slow API call.
CONVERSATIONAL_TRIGGERS = {"it", "its", "they", "them", "that", "those", "this", "these"}

REWRITE_PROMPT = """
You are an expert query analyst. Your task is to rewrite a user's latest question into a standalone question that can be understood without the context of the chat history.

**CRITICAL RULES:**
1.  **If the "Latest User Question" is already a complete, standalone question, you MUST return it exactly as it is.** Do not modify it.
2.  If the "Latest User Question" contains pronouns (like "it", "its", "they") or ambiguous references ("this drug", "that"), use the "Chat History" to resolve these references and create a complete question.
3.  Your output MUST be only the rewritten question. Do not add any commentary.

**EXAMPLE 1 (Rewrite Needed):**
- **Chat History:**
  - user: Who is the sponsor for Esketamine?
  - assistant: Janssen-Cilag Pty Ltd. is the sponsor for Esketamine.
- **Latest User Question:** what is its use?
- **Your Rewritten Question:** What is the use of Esketamine?

**EXAMPLE 2 (No Rewrite Needed):**
- **Chat History:**
  - user: Who is the sponsor for Esketamine?
  - assistant: Janssen-Cilag Pty Ltd. is the sponsor for Esketamine.
- **Latest User Question:** What is the dosage form for Fruquintinib?
- **Your Rewritten Question:** What is the dosage form for Fruquintinib?

**TASK:**
- **Chat History:**
{chat_history}
- **Latest User Question:** {question}

**Your Rewritten Question:**
"""

class QueryRewriter:
    def __init__(self):
        self.llm = get_flash_model()
        if not self.llm:
            logger.error("FATAL: Gemini client not initialized, QueryRewriter will not work.")

    # --- START OF DEFINITIVE FIX: Resilient LLM Call ---
    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """A wrapper for generate_content that handles API retries and timeouts gracefully."""
        if not self.llm:
            logger.error("LLM model for QueryRewriter is not available.")
            return None
        try:
            response = self.llm.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            return response.text
        except google_exceptions.RetryError as e:
            logger.error(f"QueryRewriter API call timed out after multiple retries: {e}")
            return None # Signal recoverable failure
        except Exception as e:
            logger.error(f"An unexpected error occurred during QueryRewriter LLM call: {e}", exc_info=True)
            return None
    # --- END OF DEFINITIVE FIX ---

    def rewrite(self, query: str, chat_history: List[str]) -> str:
        """Rewrites a conversational query into a standalone query, with a performance-enhancing pre-check."""
        if not self.llm or not chat_history:
            return query

        query_words = set(query.lower().split())
        if not CONVERSATIONAL_TRIGGERS.intersection(query_words):
            logger.info(f"Query deemed standalone. Bypassing LLM rewrite. Query: '{query}'")
            return query
        
        formatted_history = "\n  - ".join(chat_history)
        prompt = REWRITE_PROMPT.format(chat_history=formatted_history, question=query)
        
        # --- START OF DEFINITIVE FIX: Use the resilient wrapper ---
        rewritten_query_text = self._call_llm_with_retry(prompt)

        if rewritten_query_text is None:
            logger.warning("Query rewrite failed due to API issues. Using original query as fallback.")
            return query # Fallback to original query on API failure

        rewritten_query = rewritten_query_text.strip()
        # --- END OF DEFINITIVE FIX ---
        
        if rewritten_query:
            logger.info(f"Original query: '{query}' -> Rewritten query: '{rewritten_query}'")
            return rewritten_query
        else:
            logger.warning("Query rewrite resulted in an empty string. Using original query.")
            return query