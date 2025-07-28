# FILE: src/agent_v2.py (UPDATED for Phase 3.3 Exception Hardening)

import logging
from typing import List

from src.models import QueryMetadata, ToolResult
from src.planner.query_classifier import QueryClassifier
from src.planner.tool_planner import ToolPlanner
from src.planner.confidence import ToolConfidenceScorer
from src.router.tool_router import ToolRouter
from src.fallback import should_trigger_fallback, render_fallback_message
from src.middleware.logging import log_trace, Timer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

class AgentV2:
    def __init__(self, confidence_threshold: float = 0.9):
        self.classifier = QueryClassifier()
        self.planner = ToolPlanner(coverage_threshold=confidence_threshold)
        self.confidence_scorer = ToolConfidenceScorer(min_threshold=confidence_threshold)
        self.router = ToolRouter()

    def run(self, query: str) -> str:
        logger.info(f"AgentV2 received query: {query}")

        with Timer() as timer:
            try:
                query_meta = self.classifier.classify(query)
                if not query_meta:
                    return "Sorry, I couldn't understand your question. Could you rephrase it?"

                tool_plan = self.planner.plan(query_meta)
                results: List[ToolResult] = []

                for plan_item in tool_plan:
                    try:
                        tool_result = self.router.execute_tool(
                            plan_item.tool_name, query, query_meta
                        )
                    except Exception as e:
                        logger.error(f"Tool '{plan_item.tool_name}' raised unexpected error: {e}")
                        tool_result = ToolResult(
                            tool_name=plan_item.tool_name,
                            success=False,
                            content=f"Tool error: {str(e)}",
                            estimated_coverage=0.0
                        )

                    tool_result.estimated_coverage = plan_item.estimated_coverage
                    results.append(tool_result)

                    if self.confidence_scorer.has_sufficient_coverage(results):
                        logger.info("Coverage threshold met. Stopping early.")
                        break

                if should_trigger_fallback(results):
                    return render_fallback_message(query)

                response = self._synthesize_answer([r for r in results if r.success])
                return response

            finally:
                if 'query_meta' in locals():
                    log_trace(query, query_meta, tool_plan, results, timer.duration)

    def _synthesize_answer(self, results: List[ToolResult]) -> str:
        chunks = [f"[{r.tool_name.upper()}] {r.content}" for r in results if r.content]
        return "\n---\n".join(chunks)

# Note: all tool errors are now caught per-call, never propagate uncaught exceptions.