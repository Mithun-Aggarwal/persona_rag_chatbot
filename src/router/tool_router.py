# FILE: src/router/tool_router.py
# V2.0: Modular dispatcher for the unified agent.
"""
Maps tool names from the planner to callable functions from the tool library.
This module replaces hardcoded routing with a pluggable, extensible registry.
"""
import logging
from typing import Callable, Dict

from src.models import QueryMetadata, ToolResult
from src.tools import retrievers # Import the entire retrievers module

logger = logging.getLogger(__name__)

class ToolRouter:
    """
    Executes a specific tool by name, providing a unified and safe interface.
    """
    def __init__(self):
        """
        Initializes the router by creating a registry of all available tools.
        The keys MUST match the 'tool_name' in `persona_tool_map.yml` and the planner logic.
        """
        self.registry: Dict[str, Callable[[str, QueryMetadata], ToolResult]] = {
            # Vector Search Tools
            "retrieve_clinical_data": retrievers.retrieve_clinical_data,
            "retrieve_summary_data": retrievers.retrieve_summary_data,
            "retrieve_general_text": retrievers.retrieve_general_text,
            
            # Graph Search Tool
            "query_knowledge_graph": retrievers.query_knowledge_graph,
            
            # Future tools can be added here easily:
            # "search_live_web": tools.search_live_web,
        }
        logger.info(f"ToolRouter initialized with {len(self.registry)} tools: {list(self.registry.keys())}")

    def execute_tool(self, tool_name: str, query: str, query_meta: QueryMetadata) -> ToolResult:
        """
        Looks up a tool by name in the registry and executes it.

        Args:
            tool_name: The name of the tool to execute.
            query: The original user query.
            query_meta: The structured metadata about the query.

        Returns:
            A ToolResult object with the outcome of the execution.
        """
        logger.info(f"[ToolRouter] Attempting to execute tool: '{tool_name}'")
        
        tool_function = self.registry.get(tool_name)

        if not tool_function:
            logger.warning(f"[ToolRouter] Tool '{tool_name}' not found in registry. Returning a failure result.")
            return ToolResult(
                tool_name=tool_name, 
                success=False, 
                content="[Error: Tool not implemented or misconfigured]"
            )

        try:
            # Call the registered function with the required arguments
            return tool_function(query, query_meta)
        except Exception as e:
            # This is a critical safety net. If a tool fails unexpectedly,
            # it won't crash the entire agent.
            logger.error(f"[ToolRouter] Tool '{tool_name}' failed with an unhandled exception: {e}", exc_info=True)
            return ToolResult(
                tool_name=tool_name, 
                success=False, 
                content=f"An unexpected error occurred in the tool: {e}"
            )