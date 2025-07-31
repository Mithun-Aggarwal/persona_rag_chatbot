# FILE: src/agent.py
# V6.0 (Hybrid Search): Implemented concurrent tool execution and Cohere re-ranking.

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

# --- NEW IMPORTS ---
from src.tools.clients import get_generative_model, get_flash_model, get_cohere_client, DEFAULT_REQUEST_OPTIONS
from src.common.utils import load_config # Assuming you have a utils file to load yaml

from src.models import ToolResult, QueryMetadata, ToolPlanItem
from src.planner.query_classifier import QueryClassifier
from src.planner.tool_planner import ToolPlanner
from src.planner.persona_classifier import PersonaClassifier
from src.planner.query_rewriter import QueryRewriter
from src.router.tool_router import ToolRouter
from src.prompts import DECOMPOSITION_PROMPT, REASONING_SYNTHESIS_PROMPT, DIRECT_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)
LOG_PATH = Path("trace_logs.jsonl")

# Load re-ranker config
RERANKER_CONFIG = load_config("config/models.yml").get("reranker", {})

class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self): self.start = time.perf_counter(); return self
    def __exit__(self, *args): self.end = time.perf_counter(); logger.info(f"[TIMER] {self.name} took {(self.end - self.start) * 1000:.2f} ms")

def log_trace(query: str, persona: str, query_meta: QueryMetadata, tool_plan: List[ToolPlanItem], tool_results: List[ToolResult], final_answer: str, total_latency_sec: float):
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
        with open(LOG_PATH, "a", encoding="utf-8") as f: f.write(json.dumps(trace_record) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to trace log: {e}", exc_info=True)


class Agent:
    def __init__(self, confidence_threshold: float = 0.85):
        self.classifier = QueryClassifier()
        self.planner = ToolPlanner(coverage_threshold=confidence_threshold)
        self.router = ToolRouter()
        self.persona_classifier = PersonaClassifier()
        self.rewriter = QueryRewriter()
        self.llm = get_generative_model(RERANKER_CONFIG.get('../models.yml/synthesis_pro', 'gemini-1.5-pro-latest'))
        self.decomposer_llm = get_flash_model(RERANKER_CONFIG.get('../models.yml/synthesis_flash', 'gemini-1.5-flash-latest'))
        # --- NEW: Initialize Cohere client ---
        self.reranker_client = get_cohere_client()

    # --- START OF DEFINITIVE REFACTOR for Hybrid Search ---
    def _run_single_rag_step(self, query: str, persona: str) -> Tuple[str, QueryMetadata, List[ToolPlanItem], List[ToolResult]]:
        final_answer, query_meta, tool_plan, results = "", None, [], []
        
        with Timer(f"Single RAG Step for '{query[:30]}...'"):
            with Timer("Query Classification"):
                query_meta = self.classifier.classify(query)
            if not query_meta: return "I had trouble understanding the sub-question.", query_meta, tool_plan, results

            with Timer("Tool Planning"):
                tool_plan = self.planner.plan(query_meta, persona)
            if not tool_plan: return "I don't have a configured strategy for this sub-question.", query_meta, tool_plan, results

            # 1. Execute all tools concurrently
            with Timer("Concurrent Tool Execution"):
                with ThreadPoolExecutor(max_workers=len(tool_plan)) as executor:
                    futures = [executor.submit(self.router.execute_tool, item.tool_name, query, query_meta) for item in tool_plan]
                    results = [future.result() for future in futures]

            # 2. Parse all successful results into a list of documents
            all_docs = []
            successful_tool_results = [res for res in results if res and res.success and res.content.strip()]
            for res in successful_tool_results:
                # Split content by the "---" separator we use between Pinecone results
                snippets = res.content.split("\n---\n")
                all_docs.extend(snippets)

            if not all_docs:
                logger.warning("All tools ran but returned no content. Triggering fallback.")
                return "I searched but could not find any relevant details for this step.", query_meta, tool_plan, results

            # 3. Re-rank the collected documents
            ranked_docs = []
            if self.reranker_client and all_docs:
                with Timer("Re-ranking with Cohere"):
                    try:
                        rerank_response = self.reranker_client.rerank(
                            model=RERANKER_CONFIG.get("model_name", "rerank-english-v3.0"),
                            query=query,
                            documents=all_docs,
                            top_n=5 # Only take the top 5 most relevant results
                        )
                        # Reconstruct the list of documents in the new, superior order
                        ranked_docs = [all_docs[r.index] for r in rerank_response.results if r.relevance_score > 0.1]
                        logger.info(f"Re-ranked {len(all_docs)} snippets down to {len(ranked_docs)}.")
                    except Exception as e:
                        logger.error(f"Cohere re-ranking failed: {e}. Falling back to using all docs.", exc_info=True)
                        ranked_docs = all_docs # Fallback to original list on error
            else:
                ranked_docs = all_docs # If no re-ranker, use all docs

            if not ranked_docs:
                logger.warning("Re-ranking filtered out all documents. Triggering fallback.")
                return "I searched and found some initial information, but it did not seem relevant to your specific question.", query_meta, tool_plan, results

            # 4. Synthesize the final answer using the top-ranked context
            formatted_context = "\n---\n".join(ranked_docs)
            final_prompt = DIRECT_SYNTHESIS_PROMPT.format(question=query, context_str=formatted_context)
            
            with Timer("Direct Synthesis LLM Call (Flash)"):
                if not self.decomposer_llm: return "Flash model not available.", query_meta, tool_plan, results
                response = self.decomposer_llm.generate_content(final_prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            
            final_answer = response.text
            return final_answer, query_meta, tool_plan, results
    # --- END OF DEFINITIVE REFACTOR ---

    def run(self, query: str, persona: str, chat_history: List[str]) -> str:
        # This method's logic remains largely the same, as the complexity is now inside _run_single_rag_step
        run_start_time = time.perf_counter()
        final_answer, final_query_meta, final_tool_plan, final_tool_results = "", None, [], []
        try:
            with Timer("Full Agent Run"):
                if not self.llm or not self.decomposer_llm: return "Error: AI model not available."
                rewritten_query = self.rewriter.rewrite(query, chat_history)
                
                try:
                    decomp_prompt = DECOMPOSITION_PROMPT.format(chat_history="\n- ".join(chat_history), question=rewritten_query)
                    decomp_response = self.decomposer_llm.generate_content(decomp_prompt, request_options=DEFAULT_REQUEST_OPTIONS)
                    plan_data = json.loads(decomp_response.text)
                except Exception:
                    plan_data = {"requires_decomposition": False, "plan": [rewritten_query]}
                
                plan = plan_data.get("plan", [rewritten_query])

                if not plan_data.get("requires_decomposition", False) or len(plan) == 1:
                    chosen_persona = self.persona_classifier.classify(plan[0]) if persona == "automatic" else persona
                    persona_display_name = " ".join(word.capitalize() for word in chosen_persona.split("_"))
                    synthesis_result, final_query_meta, final_tool_plan, final_tool_results = self._run_single_rag_step(plan[0], chosen_persona)
                    final_answer = f"Acting as a **{persona_display_name}**, here is what I found:\n\n{synthesis_result}" if persona == "automatic" else synthesis_result
                else:
                    scratchpad = []
                    for sub_query in plan:
                        sub_persona = self.persona_classifier.classify(sub_query)
                        sub_answer, sub_meta, sub_plan, sub_results = self._run_single_rag_step(sub_query, sub_persona)
                        scratchpad.append(f"Sub-Question: {sub_query}\nFinding: {sub_answer}")
                        final_query_meta, final_tool_plan, final_tool_results = sub_meta, sub_plan, sub_results
                    synthesis_prompt = REASONING_SYNTHESIS_PROMPT.format(question=rewritten_query, scratchpad="\n\n---\n\n".join(scratchpad))
                    synthesis_response = self.llm.generate_content(synthesis_prompt, request_options=DEFAULT_REQUEST_OPTIONS)
                    final_answer = synthesis_response.text
                return final_answer
        except Exception as e:
            logger.error(f"An unexpected error occurred during agent run: {e}", exc_info=True)
            return "I encountered a critical error. Please check the system logs."
        finally:
            run_end_time = time.perf_counter()
            log_trace(query, persona, final_query_meta, final_tool_plan, final_tool_results, final_answer, run_end_time - run_start_time)