# FILE: src/planner/tool_planner.py
# Phase 1.2: ToolPlanner — scores available tools based on query intent and persona coverage

import logging
from typing import List, Tuple
from src.models import QueryMetadata, ToolPlanItem

logger = logging.getLogger(__name__)

# Hardcoded example scoring logic for Phase 1.2
# Format: (intent, tool_name) -> base score
INTENT_TOOL_SCORES = {
    ("specific_fact_lookup", "neo4j"): 0.9,
    ("specific_fact_lookup", "pinecone"): 0.6,
    ("simple_summary", "pinecone"): 0.85,
    ("simple_summary", "neo4j"): 0.4,
    ("comparative_analysis", "pinecone"): 0.8,
    ("comparative_analysis", "neo4j"): 0.6,
    ("general_qa", "pinecone"): 0.7,
    ("general_qa", "neo4j"): 0.5
}

DEFAULT_SCORE = 0.3

class ToolPlanner:
    def __init__(self, coverage_threshold: float = 0.9):
        self.coverage_threshold = coverage_threshold

    def plan(self, query_meta: QueryMetadata) -> List[ToolPlanItem]:
        """Returns a ranked list of ToolPlanItems until coverage ≥ threshold."""
        logger.info(f"Planning tools for intent: {query_meta.intent}")
        ordered_tools: List[ToolPlanItem] = []
        tools = ["neo4j", "pinecone"]

        # Assign intent-based score (later refine with persona-aware logic)
        tool_scores: List[Tuple[str, float]] = []
        for tool in tools:
            score = INTENT_TOOL_SCORES.get((query_meta.intent, tool), DEFAULT_SCORE)
            tool_scores.append((tool, score))

        # Sort by descending score
        tool_scores.sort(key=lambda x: x[1], reverse=True)

        total_coverage = 0.0
        for tool, score in tool_scores:
            if total_coverage >= self.coverage_threshold:
                break
            ordered_tools.append(ToolPlanItem(tool_name=tool, estimated_coverage=score))
            total_coverage += score

        logger.info(f"Tool plan result: {[t.model_dump() for t in ordered_tools]}")
        return ordered_tools

# Optional test block
if __name__ == "__main__":
    from src.models import QueryMetadata

    mock_meta = QueryMetadata(
        intent="specific_fact_lookup",
        keywords=["Abaloparatide", "sponsor"],
        question_is_graph_suitable=True
    )

    planner = ToolPlanner()
    plan = planner.plan(mock_meta)
    for item in plan:
        print(item.model_dump_json(indent=2))
