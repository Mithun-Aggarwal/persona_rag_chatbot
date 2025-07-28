# FILE: src/planner/confidence.py
# Phase 1.3: ToolConfidenceScorer — measures cumulative information coverage from tool results

import logging
from typing import List
from src.models import ToolResult

logger = logging.getLogger(__name__)

class ToolConfidenceScorer:
    """
    Recomputes cumulative coverage score after each tool reply based on:
    - Number of unique content blocks retrieved
    - Citation span coverage (placeholder)
    - Matching keywords (optional)
    """
    def __init__(self, min_threshold: float = 0.9):
        self.coverage_threshold = min_threshold

    def compute_total_coverage(self, results: List[ToolResult]) -> float:
        """Computes cumulative estimated coverage based on tool-level metadata."""
        total_coverage = sum([r.estimated_coverage for r in results if r.success])
        logger.info(f"Total cumulative coverage: {total_coverage:.2f}")
        return round(total_coverage, 3)

    def has_sufficient_coverage(self, results: List[ToolResult]) -> bool:
        coverage = self.compute_total_coverage(results)
        return coverage >= self.coverage_threshold

# Test block
if __name__ == "__main__":
    from src.models import ToolResult

    results = [
        ToolResult(tool_name="pinecone", estimated_coverage=0.6, success=True),
        ToolResult(tool_name="neo4j", estimated_coverage=0.35, success=True),
        ToolResult(tool_name="pdf", estimated_coverage=0.2, success=False)
    ]

    scorer = ToolConfidenceScorer()
    print("Coverage met:", scorer.has_sufficient_coverage(results))
