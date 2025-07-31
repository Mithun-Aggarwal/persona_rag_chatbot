# FILE: src/agent.py
# V5.5 (Definitive Prompt Fix): Filtered out empty tool results before synthesis to prevent LLM confusion.

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from src.tools.clients import get_generative_model, get_flash_model, DEFAULT_REQUEST_OPTIONS
from src.models import ToolResult, QueryMetadata, ToolPlanItem
from src.planner.query_classifier import QueryClassifier
from src.planner.tool_planner import ToolPlanner
from src.planner.persona_classifier import PersonaClassifier
from src.planner.query_rewriter import QueryRewriter
from src.router.tool_router import ToolRouter
from src.fallback import should_trigger_fallback
from src.prompts import DECOMPOSITION_PROMPT, REASONING_SYNTHESIS_PROMPT, DIRECT_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)
LOG_PATH = Path("trace_logs.jsonl")

class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self): self.start = time.perf_counter(); return self
    def __exit__(self, *args): self.end = time.perf_counter(); logger.info(f"[TIMER] {self.name} took {(self.end - self.start) * 1000:.2f} ms")

def log_trace(query: str, persona: str, query_meta: QueryMetadata, tool_plan: List[ToolPlanItem], tool_results: List[ToolResult], final_answer: str, total_latency_sec: float):
    # (Code unchanged)
    trace_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z", "query": query, "persona": persona,
        "intent": query_meta.intent if query_meta else "classification_failed",
        "graph_suitable": query_meta.question_is_graph_suitable if query_meta else "unknown",
        "tool_plan": [t.model_dump() for t in tool_plan] if tool_plan else [],
        "tool_results": [r.model_dump() for r in tool_results] if tool_results else [],
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
        # (Code unchanged)
        self.classifier = QueryClassifier()
        self.planner = ToolPlanner(coverage_threshold=confidence_threshold)
        self.router = ToolRouter()
        self.persona_classifier = PersonaClassifier()
        self.rewriter = QueryRewriter()
        self.llm = get_generative_model('gemini-1.5-pro-latest')
        self.decomposer_llm = get_flash_model('gemini-1.5-flash-latest')

    def _run_single_rag_step(self, query: str, persona: str) -> Tuple[str, QueryMetadata, List[ToolPlanItem], List[ToolResult]]:
        final_answer, query_meta, tool_plan, results = "", None, [], []
        with Timer(f"Single RAG Step for '{query[:30]}...'"):
            with Timer("Query Classification"):
                query_meta = self.classifier.classify(query)
            if not query_meta: return "I had trouble understanding the sub-question.", query_meta, tool_plan, results

            with Timer("Tool Planning"):
                tool_plan = self.planner.plan(query_meta, persona)
            if not tool_plan: return "I don't have a configured strategy for this sub-question.", query_meta, tool_plan, results

            results = [self.router.execute_tool(item.tool_name, query, query_meta) for item in tool_plan]

            # --- START OF DEFINITIVE FIX ---
            # The previous logic was too simple. We must ONLY use content from tools that
            # were successful AND returned non-empty content. This prevents the final
            # prompt from being "poisoned" with empty results, which confuses the LLM.
            
            successful_results = [res for res in results if res and res.success and res.content and res.content.strip()]
            
            if not successful_results:
                logger.warning("All tools ran but returned no content. Triggering fallback.")
                return "I searched but could not find any relevant details for this step.", query_meta, tool_plan, results

            successful_content = [res.content for res in successful_results]
            # --- END OF DEFINITIVE FIX ---
            
            formatted_context = "\n---\n".join(successful_content)
            final_prompt = DIRECT_SYNTHESIS_PROMPT.format(question=query, context_str=formatted_context)
            
            with Timer("Direct Synthesis LLM Call (Flash)"):
                if not self.decomposer_llm: return "Flash model not available.", query_meta, tool_plan, results
                response = self.decomposer_llm.generate_content(final_prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            
            final_answer = response.text
            return final_answer, query_meta, tool_plan, results

    def run(self, query: str, persona: str, chat_history: List[str]) -> str:
        # (Code is identical to the last correct version, no changes needed here)
        run_start_time = time.perf_counter()
        final_answer, final_query_meta, final_tool_plan, final_tool_results = "", None, [], []
        try:
            with Timer("Full Agent Run"):
                if not self.llm or not self.decomposer_llm: return "Error: AI model not available."
                with Timer("Query Rewriting"):
                    rewritten_query = self.rewriter.rewrite(query, chat_history)
                with Timer("Decomposition"):
                    try:
                        decomp_prompt = DECOMPOSITION_PROMPT.format(chat_history="\n- ".join(chat_history), question=rewritten_query)
                        decomp_response = self.decomposer_llm.generate_content(decomp_prompt, request_options=DEFAULT_REQUEST_OPTIONS)
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
                    synthesis_result, final_query_meta, final_tool_plan, final_tool_results = self._run_single_rag_step(plan[0], chosen_persona)
                    if persona == "automatic": final_answer = f"Acting as a **{persona_display_name}**, here is what I found:\n\n{synthesis_result}"
                    else: final_answer = synthesis_result
                else:
                    logger.info(f"Executing multi-step plan for query: '{rewritten_query}'")
                    scratchpad, all_sub_results = [], []
                    for sub_query in plan:
                        logger.info(f"  -> Executing sub-query: '{sub_query}'")
                        with Timer("Persona Classification (sub-query)"):
                            sub_persona = self.persona_classifier.classify(sub_query)
                        sub_answer, sub_meta, sub_plan, sub_results = self._run_single_rag_step(sub_query, sub_persona)
                        observation = f"Sub-Question: {sub_query}\nFinding: {sub_answer}"
                        scratchpad.append(observation)
                        final_query_meta, final_tool_plan = sub_meta, sub_plan
                        all_sub_results.extend(sub_results)
                    final_tool_results = all_sub_results
                    with Timer("Reasoning Synthesis LLM Call (Pro)"):
                        synthesis_prompt = REASONING_SYNTHESIS_PROMPT.format(question=rewritten_query, scratchpad="\n\n---\n\n".join(scratchpad))
                        synthesis_response = self.llm.generate_content(synthesis_prompt, request_options=DEFAULT_REQUEST_OPTIONS)
                    final_answer = synthesis_response.text
                return final_answer
        except Exception as e:
            logger.error(f"An unexpected error occurred during agent run: {e}", exc_info=True)
            return "I encountered a critical error. Please check the system logs."
        finally:
            run_end_time = time.perf_counter()
            total_duration_sec = run_end_time - run_start_time
            log_trace(query, persona, final_query_meta, final_tool_plan, final_tool_results, final_answer, total_duration_sec)