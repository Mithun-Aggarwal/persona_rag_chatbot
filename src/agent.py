# FILE: src/agent.py
# V2.5 (Definitive Fix): Simplifies the context formatting logic completely.
# It now directly joins the clean, pre-formatted output from the tools.

import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List

from src.tools.clients import get_google_ai_client
from src.models import ToolResult, QueryMetadata, ToolPlanItem
from src.planner.query_classifier import QueryClassifier
from src.planner.tool_planner import ToolPlanner
from src.router.tool_router import ToolRouter
from src.fallback import should_trigger_fallback, render_fallback_message
from src.prompts import SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)
LOG_PATH = Path("trace_logs.jsonl")

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        self.duration = -1.0
        return self
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start

def log_trace(query: str, persona: str, query_meta: QueryMetadata, tool_plan: List[ToolPlanItem], tool_results: List[ToolResult], final_answer: str, total_latency_sec: float):
    trace_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query, "persona": persona,
        "intent": query_meta.intent if query_meta else "classification_failed",
        "graph_suitable": query_meta.question_is_graph_suitable if query_meta else "unknown",
        "tool_plan": [t.model_dump() for t in tool_plan],
        "tool_results": [r.model_dump() for r in tool_results],
        "final_answer_preview": final_answer[:200] + "..." if final_answer else "N/A",
        "total_latency_sec": round(total_latency_sec, 3)
    }
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_record) + "\n")
        logger.info(f"Trace logged successfully to {LOG_PATH.resolve()}")
    except Exception as e:
        logger.error(f"Failed to write trace log: {e}", exc_info=True)

class Agent:
    def __init__(self, confidence_threshold: float = 0.85):
        self.classifier = QueryClassifier()
        self.planner = ToolPlanner(coverage_threshold=confidence_threshold)
        self.router = ToolRouter()
        genai_client = get_google_ai_client()
        self.llm = genai_client.GenerativeModel('gemini-1.5-pro-latest') if genai_client else None
        if not self.llm: logger.error("FATAL: Gemini client could not be initialized.")

    # --- START OF DEFINITIVE FIX ---
    # REMOVED the old _format_context_for_synthesis function entirely.
    # --- END OF DEFINITIVE FIX ---

    def run(self, query: str, persona: str) -> str:
        query_meta, tool_plan, results, final_answer = None, [], [], ""
        timer = Timer()
        try:
            with timer:
                if not self.llm: return "Error: AI model not available."
                query_meta = self.classifier.classify(query)
                if not query_meta: return "I'm sorry, I had trouble understanding your question."

                tool_plan = self.planner.plan(query_meta, persona)
                if not tool_plan: return "I'm sorry, I don't have a configured strategy for this persona."

                for plan_item in tool_plan:
                    results.append(self.router.execute_tool(plan_item.tool_name, query, query_meta))

                if should_trigger_fallback(results): return render_fallback_message(query)

                # --- START OF DEFINITIVE FIX ---
                # This is the new, simplified context assembly logic.
                # It filters for successful tools with real content and joins them.
                successful_content = [
                    res.content for res in results 
                    if res.success and res.content and "no relevant information" not in res.content.lower()
                ]
                formatted_context = "\n---\n".join(successful_content)
                # --- END OF DEFINITIVE FIX ---

                if not formatted_context:
                    return "I searched for information but could not find any relevant details."
                
                final_prompt = SYNTHESIS_PROMPT.format(question=query, context_str=formatted_context)
                final_answer = self.llm.generate_content(final_prompt).text
                return final_answer
        except Exception as e:
            logger.error(f"An unexpected error occurred during agent run: {e}", exc_info=True)
            final_answer = f"I encountered a critical error. Please check the system logs."
            return final_answer
        finally:
            log_trace(query, persona, query_meta, tool_plan, results, final_answer, timer.duration)