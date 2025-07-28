# FILE: src/planner/query_rewriter.py
# NEW MODULE: Implements conversational memory by rewriting user queries.

import logging
from typing import List
from src.tools.clients import get_google_ai_client

logger = logging.getLogger(__name__)

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
        genai_client = get_google_ai_client()
        if genai_client:
            self.llm = genai_client.GenerativeModel('gemini-1.5-flash-latest')
        else:
            self.llm = None
            logger.error("FATAL: Gemini client not initialized, QueryRewriter will not work.")

    def rewrite(self, query: str, chat_history: List[str]) -> str:
        """Rewrites a conversational query into a standalone query."""
        if not self.llm or not chat_history:
            return query # If no history or no LLM, cannot rewrite.

        try:
            # Format the history for the prompt
            formatted_history = "\n  - ".join(chat_history)
            prompt = REWRITE_PROMPT.format(chat_history=formatted_history, question=query)
            
            response = self.llm.generate_content(prompt)
            rewritten_query = response.text.strip()
            
            if rewritten_query:
                logger.info(f"Original query: '{query}' -> Rewritten query: '{rewritten_query}'")
                return rewritten_query
            else:
                logger.warning("Query rewrite resulted in an empty string. Using original query.")
                return query
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}", exc_info=True)
            return query # Fallback to original query on error