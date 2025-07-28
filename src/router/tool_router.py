# FILE: src/router/tool_router.py
# Phase 2.1: ToolRouter — modular dispatcher that executes tools with unified interface

import logging
from typing import Callable, Dict

from src.models import QueryMetadata, ToolResult
from src.retrievers.pinecone import run_pinecone_search
from src.retrievers.neo4j_graph import run_neo4j_search

logger = logging.getLogger(__name__)

class ToolRouter:
    """
    Maps tool names to callable functions.
    This module replaces hardcoded routing with pluggable, extensible logic.
    """

    def __init__(self):
        self.registry: Dict[str, Callable[[str, QueryMetadata], ToolResult]] = {
            "pinecone": run_pinecone_search,
            "neo4j": run_neo4j_search,
            # Future: "pdf": run_pdf_retriever,
        }

    def execute_tool(self, tool_name: str, query: str, query_meta: QueryMetadata) -> ToolResult:
        logger.info(f"[ToolRouter] Executing: {tool_name}")
        fn = self.registry.get(tool_name)

        if not fn:
            logger.warning(f"[ToolRouter] Tool '{tool_name}' not found. Returning fallback result.")
            return ToolResult(tool_name=tool_name, success=False, content="[Tool not implemented]", estimated_coverage=0.0)

        try:
            return fn(query, query_meta)
        except Exception as e:
            logger.error(f"[ToolRouter] Tool '{tool_name}' failed: {e}", exc_info=True)
            return ToolResult(tool_name=tool_name, success=False, content=f"Tool error: {e}", estimated_coverage=0.0)


# Optional test block
if __name__ == "__main__":
    from src.models import QueryMetadata

    meta = QueryMetadata(
        intent="specific_fact_lookup",
        keywords=["Abaloparatide"],
        question_is_graph_suitable=True
    )

    router = ToolRouter()
    result = router.execute_tool("neo4j", "What company sponsors Abaloparatide?", meta)
    print(result.model_dump_json(indent=2))
