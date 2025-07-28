# FILE: src/fallback.py
# Phase 3.1: FallbackLayer — graceful UX when all tools return empty or fail

import logging
from typing import List
from src.models import ToolResult

logger = logging.getLogger(__name__)

FALLBACK_QUESTIONS = [
    "Would you like to compare two drugs instead?",
    "Can I help you find a sponsor for a specific medicine?",
    "Do you want to search the original PDF documents directly?"
]


def should_trigger_fallback(results: List[ToolResult]) -> bool:
    """
    Returns True if all tools failed or produced no meaningful content.
    """
    if not results:
        logger.info("Fallback triggered: no tool results returned.")
        return True

    empty_or_failed = [r for r in results if not r.success or not r.content or len(r.content.strip()) < 5]
    if len(empty_or_failed) == len(results):
        logger.info("Fallback triggered: all tools failed or content was empty.")
        return True

    return False


def render_fallback_message(query: str) -> str:
    """
    Returns a polite fallback message with suggested next questions.
    """
    msg = f"""
I'm sorry — based on the current documents and tools, I couldn't find sufficient information to answer your question:

"{query}"

However, here are some things you can try next:

"""
    for i, q in enumerate(FALLBACK_QUESTIONS, start=1):
        msg += f"{i}. {q}\n"

    msg += "\nYou can also try rephrasing your question for better results."
    return msg.strip()
