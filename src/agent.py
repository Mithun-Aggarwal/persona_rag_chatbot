# FILE: src/agent.py
# V8.2 (Final Polish): Includes a corrected JSON parser and final presentation logic.

import json
import logging
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

from src.tools.clients import get_generative_model, get_flash_model, DEFAULT_REQUEST_OPTIONS
from src.models import ToolResult, QueryMetadata, ToolPlanItem
from src.planner.query_classifier import QueryClassifier
from src.planner.tool_planner import ToolPlanner
from src.planner.persona_classifier import PersonaClassifier
from src.planner.query_rewriter import QueryRewriter
from src.router.tool_router import ToolRouter
from src.prompts import DECOMPOSITION_PROMPT, REASONING_SYNTHESIS_PROMPT, DIRECT_SYNTHESIS_PROMPT, RERANKING_PROMPT

logger = logging.getLogger(__name__)
LOG_PATH = Path("trace_logs.jsonl")

# --- DEFINITIVE FIX: Robust JSON Parser that handles objects AND arrays ---
def extract_json_from_response(text: str) -> dict | list:
    """Finds and parses the first valid JSON object or array from a string."""
    # Regex to find JSON wrapped in markdown, supporting both objects {} and arrays []
    match = re.search(r'```json\s*([\[\{].*?[\]\}])\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        # Fallback to parsing the entire string
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON from response text: {text}")
        return {} # Return empty dict on failure


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

    def _rerank_with_gemini(self, query: str, documents: List[str]) -> List[str]:
        if not self.reranker_llm or not documents: return documents
        formatted_docs = "\n\n".join([f"DOCUMENT[{i}]:\n{doc}" for i, doc in enumerate(documents)])
        prompt = RERANKING_PROMPT.format(question=query, documents=formatted_docs)
        try:
            with Timer("Re-ranking with Gemini"):
                response = self.reranker_llm.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
                best_indices = extract_json_from_response(response.text)
                if not isinstance(best_indices, list):
                    logger.warning("Gemini re-ranker did not return a list. Falling back.")
                    return documents[:5]
                reranked_docs = [documents[i] for i in best_indices if i < len(documents)]
                logger.info(f"Re-ranked {len(documents)} snippets down to {len(reranked_docs)} using Gemini.")
                return reranked_docs
        except Exception as e:
            logger.error(f"Gemini re-ranking failed: {e}. Falling back to top 5.", exc_info=False)
            return documents[:5]

    def _run_single_rag_step(self, query: str, persona: str) -> Tuple[str, QueryMetadata, List[ToolPlanItem], List[ToolResult]]:
        with Timer(f"Single RAG Step for '{query[:30]}...'"):
            query_meta = self.classifier.classify(query)
            if not query_meta: return "I had trouble understanding the query.", None, [], []

            tool_plan = self.planner.plan(query_meta, persona)
            if not tool_plan: return "I don't have a strategy for this query.", query_meta, [], []

            with ThreadPoolExecutor(max_workers=len(tool_plan)) as executor:
                futures = [executor.submit(self.router.execute_tool, item.tool_name, query, query_meta) for item in tool_plan]
                results = [future.result() for future in futures]

            all_docs = []
            for res in results:
                if res and res.success and res.content.strip():
                    all_docs.extend(res.content.split("\n---\n"))

            if not all_docs:
                return "I searched but could not find any relevant details.", query_meta, tool_plan, results

            ranked_docs = self._rerank_with_gemini(query, all_docs)
            if not ranked_docs:
                return "I found some information, but it did not seem relevant.", query_meta, tool_plan, results

            evidence_texts, citation_links = [], []
            for doc in ranked_docs:
                parts = doc.split("\nCitation: ")
                evidence_texts.append(parts[0])
                if len(parts) > 1:
                    citation_links.append(parts[1])

            formatted_context = "\n\n".join([f"EVIDENCE [{i+1}]:\n{text}" for i, text in enumerate(evidence_texts)])
            final_prompt = DIRECT_SYNTHESIS_PROMPT.format(question=query, context_str=formatted_context)
            
            with Timer("Direct Synthesis LLM Call (Flash)"):
                response = self.synthesis_llm.generate_content(final_prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            answer_text = response.text.strip()

            used_indices = {int(m) - 1 for m in re.findall(r'\[(\d+)\]', answer_text)}
            
            final_response = answer_text
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

            return final_response, query_meta, tool_plan, results

    def run(self, query: str, persona: str, chat_history: List[str]) -> str:
        run_start_time = time.perf_counter()
        final_answer, final_query_meta, final_tool_plan, final_tool_results = "", None, [], []
        try:
            with Timer("Full Agent Run"):
                rewritten_query = self.rewriter.rewrite(query, chat_history)
                chosen_persona = self.persona_classifier.classify(rewritten_query) if persona == "automatic" else persona
                persona_display_name = " ".join(word.capitalize() for word in chosen_persona.split("_"))
                
                with Timer("Decomposition"):
                    decomp_prompt = DECOMPOSITION_PROMPT.format(chat_history="\n- ".join(chat_history), question=rewritten_query)
                    decomp_response = self.synthesis_llm.generate_content(decomp_prompt, request_options=DEFAULT_REQUEST_OPTIONS)
                    plan_data = extract_json_from_response(decomp_response.text)

                requires_decomposition = plan_data.get("requires_decomposition", False)
                plan = plan_data.get("plan", [rewritten_query])

                if not requires_decomposition or len(plan) <= 1:
                    logger.info(f"Executing single-step plan for query: '{plan[0]}'")
                    synthesis_result, final_query_meta, final_tool_plan, final_tool_results = self._run_single_rag_step(plan[0], chosen_persona)
                else:
                    logger.info(f"Executing multi-step plan for query: '{rewritten_query}'")
                    scratchpad = []
                    with ThreadPoolExecutor(max_workers=len(plan)) as executor:
                        sub_futures = [executor.submit(self._run_single_rag_step, sub_q, chosen_persona) for sub_q in plan]
                        sub_results_list = [future.result() for future in sub_futures]
                    
                    for i, sub_q in enumerate(plan):
                        sub_answer, _, _, sub_tool_results = sub_results_list[i]
                        observation = f"Sub-Question: {sub_q}\nFinding: {sub_answer}"
                        scratchpad.append(observation)
                        if sub_tool_results: final_tool_results.extend(sub_tool_results)

                    with Timer("Reasoning Synthesis LLM Call (Pro)"):
                        synthesis_prompt = REASONING_SYNTHESIS_PROMPT.format(question=rewritten_query, scratchpad="\n\n---\n\n".join(scratchpad))
                        synthesis_response = self.llm.generate_content(synthesis_prompt, request_options=DEFAULT_REQUEST_OPTIONS)
                    synthesis_result = synthesis_response.text
                
                final_answer = f"Acting as a **{persona_display_name}**, here is what I found:\n\n{synthesis_result}" if persona == "automatic" else synthesis_result
                return final_answer
        except Exception as e:
            logger.error(f"An unexpected error occurred during agent run: {e}", exc_info=True)
            return "I encountered a critical error. Please check the system logs."
        finally:
            run_end_time = time.perf_counter()
            log_trace(query, persona, final_query_meta, final_tool_plan, final_tool_results, final_answer, run_end_time - run_start_time)