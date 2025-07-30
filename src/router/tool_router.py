# FILE: src/router/tool_router.py
# V4.0 (Definitive Sync): Reverted to a simple, synchronous tool dispatcher.

import logging
from typing import Callable, Dict

from src.models import QueryMetadata, ToolResult
from src.tools import retrievers

logger = logging.getLogger(__name__)

class ToolRouter:
    def __init__(self):
        self.registry: Dict[str, Callable[[str, QueryMetadata], ToolResult]] = {
            "retrieve_clinical_data": retrievers.retrieve_clinical_data,
            "retrieve_summary_data": retrievers.retrieve_summary_data,
            "retrieve_general_text": retrievers.retrieve_general_text,
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
            # Simple, direct function call
            return tool_function(query, query_meta)
        except Exception as e:
            logger.error(f"[ToolRouter] Tool '{tool_name}' failed: {e}", exc_info=True)
            return ToolResult(tool_name=tool_name, success=False, content=f"An error occurred: {e}")