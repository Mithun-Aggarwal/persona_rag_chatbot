# FILE: src/agent.py
# V2.1: Unified Agent with Integrated Trace Logging
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

### NEW: Logging Integration - Logic from middleware is now here ###
LOG_PATH = Path("trace_logs.jsonl")

class Timer:
    """A context manager for timing code blocks."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start

def log_trace(
    query: str,
    persona: str,
    query_meta: QueryMetadata,
    tool_plan: List[ToolPlanItem],
    tool_results: List[ToolResult],
    final_answer: str,
    total_latency_sec: float
):
    """Writes a structured JSON line capturing the full agent loop."""
    trace_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "persona": persona,
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
### End of Logging Integration ###


class Agent:
    def __init__(self, confidence_threshold: float = 0.85):
        self.classifier = QueryClassifier()
        self.planner = ToolPlanner(coverage_threshold=confidence_threshold)
        self.router = ToolRouter()
        
        genai_client = get_google_ai_client()
        if genai_client:
            self.llm = genai_client.GenerativeModel('gemini-1.5-pro-latest')
        else:
            self.llm = None
            logger.error("FATAL: Gemini client could not be initialized. Agent synthesis will fail.")

    def _format_context_for_synthesis(self, results: List[ToolResult]) -> str:
        context_str = ""
        for res in results:
            if res.success and res.content and "no relevant information" not in res.content.lower():
                context_str += f"--- Evidence from Tool: {res.tool_name} ---\n"
                context_str += f"{res.content.strip()}\n\n"
        return context_str.strip()

    def run(self, query: str, persona: str) -> str:
        """Executes the full agent loop with integrated timing and logging."""
        # Initialize variables for the log trace
        query_meta, tool_plan, results, final_answer = None, [], [], ""
        
        with Timer() as timer:
            try:
                logger.info(f"Agent starting run for persona '{persona}' with query: '{query}'")

                if not self.llm:
                    return "Error: The AI model for synthesizing answers is not available. Please check API key configuration."

                query_meta = self.classifier.classify(query)
                if not query_meta:
                    final_answer = "I'm sorry, I had trouble understanding your question. Could you please rephrase it?"
                    return final_answer

                tool_plan = self.planner.plan(query_meta, persona)
                if not tool_plan:
                    final_answer = "I'm sorry, I don't have a configured strategy to answer that question for your selected persona."
                    return final_answer

                logger.info(f"Executing tool plan: {[item.tool_name for item in tool_plan]}")
                for plan_item in tool_plan:
                    results.append(self.router.execute_tool(plan_item.tool_name, query, query_meta))

                if should_trigger_fallback(results):
                    final_answer = render_fallback_message(query)
                    return final_answer

                formatted_context = self._format_context_for_synthesis(results)
                if not formatted_context:
                    final_answer = "I was able to search for information but could not find any relevant details to answer your question."
                    return final_answer
                
                final_prompt = SYNTHESIS_PROMPT.format(question=query, context_str=formatted_context)
                final_answer = self.llm.generate_content(final_prompt).text
                logger.info("Agent run completed successfully.")
                return final_answer

            except Exception as e:
                logger.error(f"An unexpected error occurred during agent run: {e}", exc_info=True)
                final_answer = f"I encountered a critical error: {e}. Please check the system logs."
                return final_answer

            finally:
                # This block ensures that a trace is logged regardless of success or failure.
                log_trace(query, persona, query_meta, tool_plan, results, final_answer, timer.duration)