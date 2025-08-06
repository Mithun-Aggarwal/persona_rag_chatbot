# FILE: src/agent.py
# V10.1 (Final Production Architecture): Implements a robust, state-aware, and
# strategic agent with graceful degradation and intelligent tool flow.
# - Prioritizes the Knowledge Graph for suitable queries, bypassing vector search
#   and re-ranking if a definitive answer is found. This fixes factual lookups.
# - Uses ThreadPoolExecutor for efficient, parallel execution of sub-queries.
# - Implements full resilience against transient API errors (e.g., 503 Overloaded).
# - Corrected the ValueError on variable unpacking.

import json
import logging
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from google.api_core import exceptions as google_exceptions
from src.tools.clients import get_generative_model, get_flash_model, DEFAULT_REQUEST_OPTIONS
from src.models import ToolResult, QueryMetadata, ToolPlanItem
from src.planner.query_classifier import QueryClassifier
from src.planner.tool_planner import ToolPlanner
from src.planner.persona_classifier import PersonaClassifier
from src.planner.query_rewriter import QueryRewriter
from src.router.tool_router import ToolRouter
from src.prompts import DECOMPOSITION_PROMPT, REASONING_SYNTHESIS_PROMPT, DIRECT_SYNTHESIS_PROMPT, RERANKING_PROMPT, SUMMARIZATION_PROMPT

logger = logging.getLogger(__name__)
LOG_PATH = Path("trace_logs.jsonl")

def extract_json_from_response(text: str) -> dict | list:
    """Finds and parses the first valid JSON object or array from a string."""
    match = re.search(r'```json\s*([\[\{].*?[\]\}])\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON from response text: {text}")
        return {}


class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self): self.start = time.perf_counter(); return self
    def __exit__(self, *args): self.end = time.perf_counter(); logger.info(f"[TIMER] {self.name} took {(self.end - self.start) * 1000:.2f} ms")

def log_trace(query: str, persona: str, query_meta: QueryMetadata, tool_plan: List[ToolPlanItem], tool_results: List[ToolResult], final_answer: str, total_latency_sec: float):
    trace_record = { "timestamp": datetime.utcnow().isoformat() + "Z", "query": query, "persona": persona, "intent": query_meta.intent if query_meta else "classification_failed", "graph_suitable": query_meta.question_is_graph_suitable if query_meta else "unknown", "tool_plan": [t.model_dump() for t in tool_plan] if tool_plan else [], "tool_results": [r.model_dump() for r in tool_results] if tool_results else [], "final_answer_preview": final_answer[:200] + "..." if final_answer else "N/A", "total_latency_sec": round(total_latency_sec, 3) }
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
        self.llm = get_generative_model('gemini-1.5-pro-latest')
        self.synthesis_llm = get_flash_model('gemini-1.5-flash-latest')
        self.reranker_llm = get_flash_model('gemini-1.5-flash-latest')
        
    def _call_llm_with_retry(self, model, prompt: str) -> Optional[str]:
        """A wrapper for generate_content that handles API retries and timeouts gracefully."""
        if not model:
            logger.error("LLM model is not available.")
            return None
        try:
            response = model.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            return response.text
        except google_exceptions.RetryError as e:
            logger.error(f"API call timed out after multiple retries: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM call: {e}", exc_info=True)
            return None

    def _rerank_with_gemini(self, query: str, documents: List[str]) -> List[str]:
        if not self.reranker_llm or not documents: return documents
        formatted_docs = "\n\n".join([f"DOCUMENT[{i}]:\n{doc}" for i, doc in enumerate(documents)])
        prompt = RERANKING_PROMPT.format(question=query, documents=formatted_docs)
        
        with Timer("Re-ranking with Gemini"):
            response_text = self._call_llm_with_retry(self.reranker_llm, prompt)
            if response_text is None:
                logger.warning("Gemini re-ranking failed due to API issues. Falling back to top 5.")
                return documents[:5]

            best_indices = extract_json_from_response(response_text)
            if not isinstance(best_indices, list):
                logger.warning("Gemini re-ranker did not return a list. Falling back.")
                return documents[:5]
            
            reranked_docs = [documents[i] for i in best_indices if i < len(documents)]
            logger.info(f"Re-ranked {len(documents)} snippets down to {len(reranked_docs)} using Gemini.")
            return reranked_docs

    def _run_single_rag_step(self, query: str, persona: str) -> Tuple[str, QueryMetadata, List[ToolPlanItem], List[ToolResult]]:
        with Timer(f"Single RAG Step for '{query[:30]}...'"):
            query_meta = self.classifier.classify(query)
            if not query_meta: return "I had trouble understanding the query.", None, [], []

            tool_plan = self.planner.plan(query_meta, persona)
            if not tool_plan: return "I don't have a strategy for this query.", query_meta, [], []
            
            # --- START OF DEFINITIVE FIX: Intelligent, Conditional Tool Flow ---
            final_results = []
            kg_success = False
            # Step 1: Prioritize the Knowledge Graph if it's suitable and planned
            if query_meta.question_is_graph_suitable and any(t.tool_name == "query_knowledge_graph" for t in tool_plan):
                kg_result = self.router.execute_tool("query_knowledge_graph", query, query_meta)
                final_results.append(kg_result)
                # Step 2: Evaluate the KG result. If it's a "golden" answer, we are done with retrieval.
                if kg_result.success and kg_result.content and len(kg_result.content.strip()) > 10:
                    logger.info("Knowledge Graph provided a definitive answer. Bypassing vector search and re-ranking.")
                    kg_success = True

            # Step 3: Fallback to Vector Search ONLY if the KG failed or wasn't suitable
            if not kg_success and any(t.tool_name == "vector_search" for t in tool_plan):
                vector_result = self.router.execute_tool("vector_search", query, query_meta)
                final_results.append(vector_result)
            # --- END OF DEFINITIVE FIX ---

            all_docs = []
            for res in final_results:
                if res and res.success and res.content and res.content.strip():
                    splitter = "\n---\n" if res.tool_name == "vector_search" else "\n"
                    all_docs.extend(res.content.split(splitter))
            
            all_docs = [doc for doc in all_docs if doc and doc.strip()]

            if not all_docs:
                return "I searched but could not find any relevant details.", query_meta, tool_plan, final_results

            if not kg_success:
                ranked_docs = self._rerank_with_gemini(query, all_docs)
            else:
                ranked_docs = all_docs
                
            if not ranked_docs:
                return "I found some information, but it did not seem relevant.", query_meta, tool_plan, final_results
            
            evidence_texts, citation_links = [], []
            for doc in ranked_docs:
                parts = doc.split("\nCitation: ")
                evidence_texts.append(parts[0])
                if len(parts) > 1:
                    citation_links.append(parts[1])

            formatted_context = "\n\n".join([f"EVIDENCE [{i+1}]:\n{text}" for i, text in enumerate(evidence_texts)])
            
            if query_meta.intent == "simple_summary":
                synthesis_prompt = SUMMARIZATION_PROMPT.format(context_str=formatted_context)
            else:
                synthesis_prompt = DIRECT_SYNTHESIS_PROMPT.format(question=query, context_str=formatted_context)
            
            with Timer("Synthesis LLM Call (Flash)"):
                answer_text = self._call_llm_with_retry(self.synthesis_llm, synthesis_prompt)
            
            if answer_text is None:
                raise google_exceptions.RetryError("Synthesis LLM failed due to API overload.", cause=None)
            
            final_response = answer_text.strip()
            if query_meta.intent != "simple_summary":
                used_indices = {int(m) - 1 for m in re.findall(r'\[(\d+)\]', answer_text)}
                if used_indices:
                    references_section = "\n\n**References**\n"
                    unique_used_links = set()
                    used_links_ordered = []
                    for idx in sorted(list(used_indices)):
                        if idx < len(citation_links):
                            link = citation_links[idx]
                            if link not in unique_used_links:
                                unique_used_links.add(link)
                                used_links_ordered.append(link)
                    references_section += "\n".join([f"{i+1}. {link}" for i, link in enumerate(used_links_ordered)])
                    final_response += references_section

            return final_response, query_meta, tool_plan, final_results

    def run(self, query: str, persona: str, chat_history: List[str]) -> str:
        run_start_time = time.perf_counter()
        final_answer, final_query_meta, final_tool_plan, final_tool_results = "", None, [], []
        try:
            with Timer("Full Agent Run"):
                rewritten_query = self.rewriter.rewrite(query, chat_history)
                
                chosen_persona_key = self.persona_classifier.classify(rewritten_query)
                if chosen_persona_key is None:
                    logger.warning("Persona classification failed due to API error. Defaulting to 'regulatory_specialist'.")
                    chosen_persona_key = "regulatory_specialist"
                chosen_persona = chosen_persona_key if persona == "automatic" else persona

                persona_display_name = " ".join(word.capitalize() for word in chosen_persona.split("_"))
                
                with Timer("Decomposition"):
                    decomp_prompt = DECOMPOSITION_PROMPT.format(chat_history="\n- ".join(chat_history), question=rewritten_query)
                    decomp_response_text = self._call_llm_with_retry(self.synthesis_llm, decomp_prompt)
                
                if decomp_response_text is None:
                    logger.warning("Decomposition failed due to API error. Degrading to single-step plan.")
                    plan_data = {"requires_decomposition": False, "plan": [rewritten_query]}
                else:
                    plan_data = extract_json_from_response(decomp_response_text)

                requires_decomposition = plan_data.get("requires_decomposition", False)
                plan = plan_data.get("plan", [rewritten_query])

                if not requires_decomposition or len(plan) <= 1:
                    logger.info(f"Executing single-step plan for query: '{plan[0]}'")
                    synthesis_result, final_query_meta, final_tool_plan, final_tool_results = self._run_single_rag_step(plan[0], chosen_persona)
                else:
                    logger.info(f"Executing multi-step plan for query: '{rewritten_query}'")
                    scratchpad = []
                    
                    retrieval_steps = []
                    logic_instruction = ""
                    LOGIC_KEYWORDS = ["identify", "compare", "contrast", "common", "difference", "both"]
                    
                    for sub_q in plan:
                        if any(keyword in sub_q.lower() for keyword in LOGIC_KEYWORDS):
                            logic_instruction = sub_q
                        else:
                            retrieval_steps.append(sub_q)
                    
                    if not retrieval_steps and logic_instruction:
                        retrieval_steps = plan

                    with ThreadPoolExecutor(max_workers=len(retrieval_steps) or 1) as executor:
                        sub_futures = [executor.submit(self._run_single_rag_step, sub_q, chosen_persona) for sub_q in retrieval_steps]
                        sub_results_list = [future.result() for future in sub_futures]
                    
                    # --- START OF DEFINITIVE FIX: Corrected result unpacking ---
                    for i, sub_q in enumerate(retrieval_steps):
                        # The function returns 4 items, we need to unpack all of them
                        sub_answer, _, _, sub_tool_results = sub_results_list[i]
                        observation = f"Observation for the question '{sub_q}':\n{sub_answer}"
                        scratchpad.append(observation)
                        if sub_tool_results: final_tool_results.extend(sub_tool_results)
                    # --- END OF DEFINITIVE FIX ---

                    if logic_instruction and logic_instruction not in retrieval_steps:
                        scratchpad.append(f"Final Instruction: {logic_instruction}")

                    with Timer("Reasoning Synthesis LLM Call (Pro)"):
                        synthesis_prompt = REASONING_SYNTHESIS_PROMPT.format(question=rewritten_query, scratchpad="\n\n---\n\n".join(scratchpad))
                        synthesis_result = self._call_llm_with_retry(self.llm, synthesis_prompt)
                        if synthesis_result is None:
                           raise google_exceptions.RetryError("Final synthesis LLM failed due to API overload.", cause=None)

                final_answer = f"Acting as a **{persona_display_name}**, here is what I found:\n\n{synthesis_result}" if persona == "automatic" else synthesis_result
                return final_answer
        except google_exceptions.RetryError as e:
            logger.error(f"A recoverable API error occurred and timed out: {e}", exc_info=False)
            return "I'm sorry, the AI models I rely on are currently experiencing high traffic. Please try your question again in a few moments."
        except Exception as e:
            logger.error(f"An unexpected error occurred during agent run: {e}", exc_info=True)
            return "I encountered a critical error processing your request. Please check the system logs for more details."
        finally:
            run_end_time = time.perf_counter()
            log_trace(query, persona, final_query_meta, final_tool_plan, final_tool_results, final_answer, run_end_time - run_start_time)