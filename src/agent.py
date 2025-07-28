# FILE: src/agent.py
# V2.7: Integrated QueryRewriter for conversational memory.

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
from src.planner.persona_classifier import PersonaClassifier
from src.planner.query_rewriter import QueryRewriter # NEW IMPORT
from src.router.tool_router import ToolRouter
from src.fallback import should_trigger_fallback, render_fallback_message
from src.prompts import SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)
LOG_PATH = Path("trace_logs.jsonl")

# ... (Timer class and log_trace function are unchanged) ...
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
        self.persona_classifier = PersonaClassifier()
        self.rewriter = QueryRewriter() # NEW: Initialize the query rewriter
        genai_client = get_google_ai_client()
        self.llm = genai_client.GenerativeModel('gemini-1.5-pro-latest') if genai_client else None
        if not self.llm: logger.error("FATAL: Gemini client could not be initialized.")

    def run(self, query: str, persona: str, chat_history: List[str]) -> str: # MODIFIED: Accept chat_history
        query_meta, tool_plan, results, final_answer = None, [], [], ""
        timer = Timer()
        try:
            with timer:
                if not self.llm: return "Error: AI model not available."
                
                # --- START: New Query Rewriting Step ---
                rewritten_query = self.rewriter.rewrite(query, chat_history)
                # --- END: New Step ---
                
                chosen_persona = persona
                persona_display_name = " ".join(word.capitalize() for word in persona.split("_"))
                
                if persona == "automatic":
                    # Use the rewritten query for classification
                    chosen_persona = self.persona_classifier.classify(rewritten_query)
                    persona_display_name = " ".join(word.capitalize() for word in chosen_persona.split("_"))
                    logger.info(f"Automatic persona selected: {chosen_persona}")

                # Use the rewritten query for the rest of the pipeline
                query_meta = self.classifier.classify(rewritten_query)
                if not query_meta: return "I'm sorry, I had trouble understanding your question."

                tool_plan = self.planner.plan(query_meta, chosen_persona)
                if not tool_plan: return f"I'm sorry, I don't have a configured strategy for a **{persona_display_name}** for this question."

                for plan_item in tool_plan:
                    results.append(self.router.execute_tool(plan_item.tool_name, rewritten_query, query_meta))

                if should_trigger_fallback(results): return render_fallback_message(rewritten_query)

                successful_content = [res.content for res in results if res.success and res.content]
                formatted_context = "\n---\n".join(successful_content)

                if not formatted_context:
                    return "I searched for information but could not find any relevant details."
                
                final_prompt = SYNTHESIS_PROMPT.format(question=rewritten_query, context_str=formatted_context)
                synthesis_result = self.llm.generate_content(final_prompt).text
                
                if persona == "automatic":
                    final_answer = f"Acting as a **{persona_display_name}**, here is what I found:\n\n{synthesis_result}"
                else:
                    final_answer = synthesis_result

                return final_answer
        except Exception as e:
            logger.error(f"An unexpected error occurred during agent run: {e}", exc_info=True)
            final_answer = f"I encountered a critical error. Please check the system logs."
            return final_answer
        finally:
            # Log the original query for traceability
            log_trace(query, persona, query_meta, tool_plan, results, final_answer, timer.duration)