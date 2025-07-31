# FILE: src/router/tool_router.py
# V5.0 (Unified Tooling): Refactored to use only the two primary tools.

import logging
from typing import Callable, Dict

from src.models import QueryMetadata, ToolResult
from src.tools import retrievers

logger = logging.getLogger(__name__)

class ToolRouter:
    def __init__(self):
        # --- DEFINITIVE FIX: Register only the tools that now exist ---
        self.registry: Dict[str, Callable[[str, QueryMetadata], ToolResult]] = {
            "vector_search": retrievers.vector_search,
            "query_knowledge_graph": retrievers.query_knowledge_graph,
        }
        logger.info(f"ToolRouter initialized with {len(self.registry)} tools.")

    def execute_tool(self, tool_name: str, query: str, query_meta: QueryMetadata) -> ToolResult:
        logger.info(f"[ToolRouter] Executing tool: '{tool_name}'")
        tool_function = self.registry.get(tool_name)
        if not tool_function:
            logger.warning(f"Tool '{tool_name}' not found in registry.")
            return ToolResult(tool_name=tool_name, success=False, content="[Error: Tool not implemented]")
        try:
            return tool_function(query, query_meta)
        except Exception as e:
            logger.error(f"[ToolRouter] Tool '{tool_name}' failed: {e}", exc_info=True)
            return ToolResult(tool_name=tool_name, success=False, content=f"An error occurred: {e}")