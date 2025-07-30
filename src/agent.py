# FILE: src/agent.py
# V5.1 (Definitive Fix): Corrected the total latency logging to align with the new Timer class.

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from src.tools.clients import get_google_ai_client
from src.models import ToolResult, QueryMetadata, ToolPlanItem
from src.planner.query_classifier import QueryClassifier
from src.planner.tool_planner import ToolPlanner
from src.planner.persona_classifier import PersonaClassifier
from src.planner.query_rewriter import QueryRewriter
from src.router.tool_router import ToolRouter
from src.fallback import should_trigger_fallback, render_fallback_message
from src.prompts import DECOMPOSITION_PROMPT, REASONING_SYNTHESIS_PROMPT, DIRECT_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)
LOG_PATH = Path("trace_logs.jsonl")

class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.end = time.perf_counter()
        duration_ms = (self.end - self.start) * 1000
        logger.info(f"[TIMER] {self.name} took {duration_ms:.2f} ms")

def log_trace(query: str, persona: str, query_meta: QueryMetadata, tool_plan: List[ToolPlanItem], tool_results: List[ToolResult], final_answer: str, total_latency_sec: float):
    trace_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z", "query": query, "persona": persona,
        "intent": query_meta.intent if query_meta else "classification_failed",
        "graph_suitable": query_meta.question_is_graph_suitable if query_meta else "unknown",
        "tool_plan": [t.model_dump() for t in tool_plan], "tool_results": [r.model_dump() for r in tool_results],
        "final_answer_preview": final_answer[:200] + "..." if final_answer else "N/A",
        "total_latency_sec": round(total_latency_sec, 3)
    }
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_record) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to trace log: {e}", exc_info=True)

class Agent:
    def __init__(self, confidence_threshold: float = 0.85):
        self.classifier = QueryClassifier()
        self.planner = ToolPlanner(coverage_threshold=confidence_threshold)
        self.router = ToolRouter()
        self.persona_classifier = PersonaClassifier()
        self.rewriter = QueryRewriter()
        genai_client = get_google_ai_client()
        self.llm = genai_client.GenerativeModel('gemini-1.5-pro-latest') if genai_client else None
        self.decomposer_llm = genai_client.GenerativeModel('gemini-1.5-flash-latest') if genai_client else None

    def _run_single_rag_step(self, query: str, persona: str) -> str:
        with Timer(f"Single RAG Step for '{query[:30]}...'"):
            with Timer("Query Classification"):
                query_meta = self.classifier.classify(query)
            if not query_meta: return "I had trouble understanding the sub-question."

            with Timer("Tool Planning"):
                tool_plan = self.planner.plan(query_meta, persona)
            if not tool_plan: return "I don't have a configured strategy for this sub-question."

            results = [self.router.execute_tool(item.tool_name, query, query_meta) for item in tool_plan]

            if should_trigger_fallback(results): return "I searched but could not find any relevant details for this step."
            
            successful_content = [res.content for res in results if res.success and res.content]
            if not successful_content: return "I found no specific details for this sub-question."
            
            formatted_context = "\n---\n".join(successful_content)
            final_prompt = DIRECT_SYNTHESIS_PROMPT.format(question=query, context_str=formatted_context)
            with Timer("Direct Synthesis LLM Call"):
                response = self.llm.generate_content(final_prompt)
            return response.text

    def run(self, query: str, persona: str, chat_history: List[str]) -> str:
        # --- START: Definitive Fix ---
        run_start_time = time.perf_counter()
        final_answer = ""
        query_meta, tool_plan, results = None, [], [] # Keep for logging
        
        try:
            with Timer("Full Agent Run"):
                if not self.llm: return "Error: AI model not available."
                
                with Timer("Query Rewriting"):
                    rewritten_query = self.rewriter.rewrite(query, chat_history)

                with Timer("Decomposition"):
                    try:
                        decomp_prompt = DECOMPOSITION_PROMPT.format(chat_history="\n- ".join(chat_history), question=rewritten_query)
                        decomp_response = self.decomposer_llm.generate_content(decomp_prompt)
                        if not decomp_response.parts: raise ValueError("LLM returned empty/blocked response")
                        plan_data = json.loads(decomp_response.text)
                    except (json.JSONDecodeError, ValueError, Exception) as e:
                        logger.warning(f"Could not decompose query: {e}. Defaulting to single-step plan.")
                        plan_data = {"requires_decomposition": False, "plan": [rewritten_query]}
                
                requires_decomposition = plan_data.get("requires_decomposition", False)
                plan = plan_data.get("plan", [rewritten_query])

                if not requires_decomposition or len(plan) == 1:
                    logger.info(f"Executing single-step plan for query: '{plan[0]}'")
                    with Timer("Persona Classification"):
                        chosen_persona = self.persona_classifier.classify(plan[0]) if persona == "automatic" else persona
                    persona_display_name = " ".join(word.capitalize() for word in chosen_persona.split("_"))
                    synthesis_result = self._run_single_rag_step(plan[0], chosen_persona)
                    if persona == "automatic":
                        final_answer = f"Acting as a **{persona_display_name}**, here is what I found:\n\n{synthesis_result}"
                    else:
                        final_answer = synthesis_result
                else:
                    logger.info(f"Executing multi-step plan for query: '{rewritten_query}'")
                    scratchpad = []
                    for sub_query in plan:
                        logger.info(f"  -> Executing sub-query: '{sub_query}'")
                        with Timer("Persona Classification (sub-query)"):
                            sub_persona = self.persona_classifier.classify(sub_query)
                        sub_answer = self._run_single_rag_step(sub_query, sub_persona)
                        observation = f"Sub-Question: {sub_query}\nFinding: {sub_answer}"
                        scratchpad.append(observation)
                    
                    with Timer("Reasoning Synthesis LLM Call"):
                        synthesis_prompt = REASONING_SYNTHESIS_PROMPT.format(question=rewritten_query, scratchpad="\n\n---\n\n".join(scratchpad))
                        synthesis_response = self.llm.generate_content(synthesis_prompt)
                    final_answer = synthesis_response.text
                
                return final_answer

        except Exception as e:
            logger.error(f"An unexpected error occurred during agent run: {e}", exc_info=True)
            final_answer = "I encountered a critical error. Please check the system logs."
            return final_answer
        finally:
            run_end_time = time.perf_counter()
            total_duration_sec = run_end_time - run_start_time
            log_trace(query, persona, query_meta, tool_plan, results, final_answer, total_duration_sec)
        # --- END: Definitive Fix ---