# src/agent.py

"""
The Core Orchestrator: MainAgent (V1.2)

This module defines the MainAgent class.
V1.2:
- Expands the graph schema provided to the Cypher generation prompt for higher accuracy.
- Fixes the citation formatter to correctly handle lists of page numbers.
"""

import streamlit as st
import google.generativeai as genai
import logging
from typing import List, Dict, Any

from src import prompts
from src import tools
from src.routing.persona_router import PersonaRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

class MainAgent:
    def __init__(self, persona: str):
        self.persona = persona
        self.router = PersonaRouter()
        try:
            self.llm = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("Gemini 1.5 Flash model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            st.error("Could not initialize the AI model. Please check API key and configuration.")
            self.llm = None

    def _get_intent(self, query: str) -> str:
        logger.info("Step 1: Analyzing user intent.")
        prompt = prompts.INTENT_ANALYSIS_PROMPT.format(query=query)
        try:
            response = self.llm.generate_content(prompt)
            intent = response.text.strip().lower()
            logger.info(f"Intent classified as: '{intent}'")
            return intent
        except Exception as e:
            logger.error(f"Error during intent analysis: {e}")
            return "general_qa"

    def _generate_cypher(self, query: str) -> str | None:
        """Generates a Cypher query from the user's question if needed."""
        logger.info("Generating Cypher query for graph search.")
        
        # --- FIX #1: Provide a much more detailed and accurate graph schema ---
        graph_schema = """
        Node labels and their properties:
        - Drug: {name: string, trade_name: string}
        - Sponsor: {name: string}
        - Disease: {name: string}
        - Indication: {name: string}
        - SubmissionType: {name: string}

        Relationship types:
        - (Drug)-[:HAS_SPONSOR]->(Sponsor)
        - (Drug)-[:HAS_INDICATION]->(Indication)
        - (Drug)-[:HAS_SUBMISSION_TYPE]->(SubmissionType)
        - (Drug)-[:HAS_TRADE_NAME]->(string)
        """
        
        prompt = prompts.CYPHER_GENERATION_PROMPT.format(schema=graph_schema, question=query)
        try:
            response = self.llm.generate_content(prompt)
            cypher_query = response.text.strip().replace("`", "").replace("cypher", "") # Clean up markdown
            if "ERROR" in cypher_query or "MATCH" not in cypher_query.upper():
                logger.warning(f"LLM could not generate a valid Cypher query. Response: {cypher_query}")
                return None
            logger.info(f"Generated Cypher: {cypher_query}")
            return cypher_query
        except Exception as e:
            logger.error(f"Error during Cypher generation: {e}")
            return None

    def _format_context_for_prompt(self, context: List[Dict[str, Any]]) -> str:
        """Formats the retrieved context into a string for the synthesis prompt."""
        logger.info("Formatting retrieved context for final synthesis.")
        context_str = ""
        for i, item in enumerate(context):
            source_info = item.get('source', {})
            content = item.get('content', 'No content available.')
            
            citation = "N/A"
            doc_id = source_info.get("document_id", "N/A")
            
            # --- FIX #2: Correctly handle page_numbers (list) vs page_number (singular) ---
            pages = source_info.get("page_numbers") or [source_info.get("page_number")]
            if pages and all(p is not None for p in pages):
                # Format the list of pages nicely
                page_str = ", ".join(map(str, sorted(list(set(pages)))))
                citation = f"[doc: {doc_id}, page: {page_str}]"
            else:
                citation = f"[doc: {doc_id}]"

            context_str += f"--- Context Source {i+1} ---\n"
            context_str += f"Source: {citation}\n"
            context_str += f"Content: {content}\n\n"
        
        return context_str

    def _synthesize_answer(self, query: str, context_str: str) -> str:
        # ... (This function remains unchanged) ...
        logger.info("Step 4: Synthesizing final answer from context.")
        prompt = prompts.SYNTHESIS_PROMPT.format(
            persona=self.persona,
            question=query,
            context_str=context_str
        )
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error during final answer synthesis: {e}")
            return "I apologize, but I encountered an error while trying to formulate a response."

    def run(self, query: str) -> str:
        # ... (This function remains largely unchanged) ...
        if not self.llm: return "The AI model is not available. Please check the configuration."
        
        intent = self._get_intent(query)
        namespaces_to_search = self.router.get_retrieval_plan(self.persona)
        logger.info(f"Plan: Search namespaces {namespaces_to_search} with intent '{intent}'")

        retrieved_context = []
        for namespace in namespaces_to_search:
            pinecone_results = tools.pinecone_search_tool(query=query, namespace=namespace)
            if pinecone_results: retrieved_context.extend(pinecone_results)
        
        if intent == "graph_query":
            cypher_query = self._generate_cypher(query)
            if cypher_query:
                graph_results = tools.neo4j_graph_tool(cypher_query=cypher_query)
                if graph_results: retrieved_context.extend(graph_results)

        if not retrieved_context:
            return "Based on the provided documents and knowledge graph, I could not find any relevant information to answer this question."

        formatted_context = self._format_context_for_prompt(retrieved_context)
        final_answer = self._synthesize_answer(query, formatted_context)
        logger.info("Agent run completed.")
        return final_answer