# src/agent.py

"""
The Core Orchestrator: MainAgent (V1.1)

This module defines the MainAgent class, which acts as the "first brain" of our RAG system.
V1.1: Upgraded to a newer Gemini model and added detailed DEBUG-level logging for prompts/responses.
"""

import streamlit as st
import google.generativeai as genai
import logging
from typing import List, Dict, Any

from src import prompts
from src import tools
from src.routing.persona_router import PersonaRouter

# --- Configure logging to show DEBUG level messages ---
# This will make our new, detailed logs visible in the console.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__) # Use a dedicated logger for this module

class MainAgent:
    """
    The main orchestrator agent.
    """
    def __init__(self, persona: str):
        self.persona = persona
        self.router = PersonaRouter()

        try:
            # --- FIX: Upgraded to a newer, more stable Gemini model ---
            # Using gemini-1.5-flash is faster and more cost-effective.
            self.llm = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("Gemini 1.5 Flash model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            st.error("Could not initialize the AI model. Please check API key and configuration.")
            self.llm = None

    def _get_intent(self, query: str) -> str:
        """Uses an LLM call to determine the user's intent."""
        logger.info("Step 1: Analyzing user intent.")
        prompt = prompts.INTENT_ANALYSIS_PROMPT.format(query=query)
        
        # --- LOGGING: Show the exact prompt being sent ---
        logger.debug(f"--- INTENT ANALYSIS PROMPT ---\n{prompt}")
        
        try:
            response = self.llm.generate_content(prompt)
            intent = response.text.strip().lower()
            
            # --- LOGGING: Show the raw response and the final intent ---
            logger.debug(f"--- INTENT ANALYSIS RAW RESPONSE ---\n{response.text}")
            logger.info(f"Intent classified as: '{intent}'")
            return intent
        except Exception as e:
            logger.error(f"Error during intent analysis: {e}")
            return "general_qa"

    def _generate_cypher(self, query: str) -> str | None:
        """Generates a Cypher query from the user's question if needed."""
        logger.info("Generating Cypher query for graph search.")
        
        graph_schema = "Node labels: 'Drug', 'Disease', 'MechanismOfAction'; Relationship types: 'INDICATED_FOR', 'HAS_MECHANISM_OF_ACTION'; Drug properties: 'name' (string); Disease properties: 'name' (string)"
        
        prompt = prompts.CYPHER_GENERATION_PROMPT.format(schema=graph_schema, question=query)
        
        # --- LOGGING: Show the exact prompt being sent ---
        logger.debug(f"--- CYPHER GENERATION PROMPT ---\n{prompt}")
        
        try:
            response = self.llm.generate_content(prompt)
            cypher_query = response.text.strip()

            # --- LOGGING: Show the raw response ---
            logger.debug(f"--- CYPHER GENERATION RAW RESPONSE ---\n{cypher_query}")

            if "ERROR" in cypher_query or "MATCH" not in cypher_query:
                logger.warning("LLM could not generate a valid Cypher query.")
                return None
            logger.info(f"Generated Cypher: {cypher_query}")
            return cypher_query
        except Exception as e:
            logger.error(f"Error during Cypher generation: {e}")
            return None

    def _format_context_for_prompt(self, context: List[Dict[str, Any]]) -> str:
        """Formats the retrieved context into a string for the synthesis prompt."""
        logger.info("Formatting retrieved context for final synthesis.")
        # ... (This function remains unchanged)
        context_str = ""
        for i, item in enumerate(context):
            source_info = item.get('source', {})
            content = item.get('content', 'No content available.')
            citation = "N/A"
            source_type = source_info.get("type")
            if source_type == "graph_path":
                primary_src = source_info.get("primary_source", {})
                citation = f"[graph: {primary_src.get('type', 'UNKNOWN_REL')}]"
            elif "document_id" in source_info:
                doc_id = source_info.get("document_id", "N/A")
                page = source_info.get("page_number", "N/A")
                citation = f"[doc: {doc_id}, page: {page}]"

            context_str += f"--- Context Source {i+1} ---\n"
            context_str += f"Source: {citation}\n"
            context_str += f"Content: {content}\n\n"
        return context_str

    def _synthesize_answer(self, query: str, context_str: str) -> str:
        """Uses an LLM call to synthesize the final answer from the context."""
        logger.info("Step 4: Synthesizing final answer from context.")
        prompt = prompts.SYNTHESIS_PROMPT.format(
            persona=self.persona,
            question=query,
            context_str=context_str
        )

        # --- LOGGING: Show the massive final prompt being sent ---
        logger.debug(f"--- FINAL SYNTHESIS PROMPT ---\n{prompt}")

        try:
            response = self.llm.generate_content(prompt)
            # --- LOGGING: Show the final raw response from the LLM ---
            logger.debug(f"--- FINAL SYNTHESIS RAW RESPONSE ---\n{response.text}")
            return response.text
        except Exception as e:
            logger.error(f"Error during final answer synthesis: {e}")
            return "I apologize, but I encountered an error while trying to formulate a response."


    def run(self, query: str) -> str:
        if not self.llm:
            return "The AI model is not available. Please check the configuration."
        
        intent = self._get_intent(query)
        
        logger.info(f"Step 2: Getting retrieval plan for persona '{self.persona}'.")
        namespaces_to_search = self.router.get_retrieval_plan(self.persona)
        logger.info(f"Plan: Search namespaces {namespaces_to_search}")

        logger.info("Step 3: Dispatching tasks to specialist tools.")
        retrieved_context = []
        
        for namespace in namespaces_to_search:
            pinecone_results = tools.pinecone_search_tool(query=query, namespace=namespace)
            if pinecone_results:
                retrieved_context.extend(pinecone_results)
        
        if intent == "graph_query":
            cypher_query = self._generate_cypher(query)
            if cypher_query:
                graph_results = tools.neo4j_graph_tool(cypher_query=cypher_query)
                if graph_results:
                    retrieved_context.extend(graph_results)

        if not retrieved_context:
            logger.warning("No context was retrieved from any data source.")
            return "Based on the provided documents and knowledge graph, I could not find any relevant information to answer this question."

        formatted_context = self._format_context_for_prompt(retrieved_context)
        
        final_answer = self._synthesize_answer(query, formatted_context)
        
        logger.info("Agent run completed.")
        return final_answer