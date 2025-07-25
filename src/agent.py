# src/agent.py

"""
The Core Orchestrator: MainAgent

This module defines the MainAgent class, which acts as the "first brain" of our RAG system.
It orchestrates the entire process from receiving a user query to generating a final,
cited answer.
"""

import streamlit as st
import google.generativeai as genai
import logging
from typing import List, Dict, Any

# Import our custom modules
from src import prompts
from src import tools
from src.routing.persona_router import PersonaRouter # Assumes this file exists as specified

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MainAgent:
    """
    The main orchestrator agent.

    This agent manages the flow of a query from the user through the two-brain system:
    1.  Analyzes the user's intent.
    2.  Gets a retrieval strategy based on the user's persona.
    3.  Dispatches queries to the specialist tools (the "second brain").
    4.  Synthesizes the retrieved context into a final answer.
    """
    def __init__(self, persona: str):
        """
        Initializes the agent with a specific user persona.

        Args:
            persona (str): The persona the agent should adopt (e.g., 'clinical_analyst').
        """
        self.persona = persona
        self.router = PersonaRouter() # Initializes the persona-to-namespace router

        # Configure the Generative AI model
        try:
            self.llm = genai.GenerativeModel('gemini-1.0-pro')
            logging.info("Gemini Pro model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini Pro model: {e}")
            st.error("Could not initialize the AI model. Please check API key and configuration.")
            self.llm = None

    def _get_intent(self, query: str) -> str:
        """Uses an LLM call to determine the user's intent."""
        logging.info("Step 1: Analyzing user intent.")
        prompt = prompts.INTENT_ANALYSIS_PROMPT.format(query=query)
        try:
            response = self.llm.generate_content(prompt)
            intent = response.text.strip().lower()
            logging.info(f"Intent classified as: '{intent}'")
            return intent
        except Exception as e:
            logging.error(f"Error during intent analysis: {e}")
            return "general_qa" # Fallback intent

    def _generate_cypher(self, query: str) -> str | None:
        """Generates a Cypher query from the user's question if needed."""
        logging.info("Generating Cypher query for graph search.")
        
        # In a production system, this schema would be dynamically fetched.
        # For now, we use a placeholder that matches our prompt examples.
        graph_schema = """
        Node labels: 'Drug', 'Disease', 'MechanismOfAction'
        Relationship types: 'INDICATED_FOR', 'HAS_MECHANISM_OF_ACTION'
        Drug properties: 'name' (string)
        Disease properties: 'name' (string)
        MechanismOfAction properties: 'name' (string)
        """
        
        prompt = prompts.CYPHER_GENERATION_PROMPT.format(schema=graph_schema, question=query)
        try:
            response = self.llm.generate_content(prompt)
            cypher_query = response.text.strip()
            if "ERROR" in cypher_query or "MATCH" not in cypher_query:
                logging.warning("LLM could not generate a valid Cypher query.")
                return None
            logging.info(f"Generated Cypher: {cypher_query}")
            return cypher_query
        except Exception as e:
            logging.error(f"Error during Cypher generation: {e}")
            return None

    def _format_context_for_prompt(self, context: List[Dict[str, Any]]) -> str:
        """Formats the retrieved context into a string for the synthesis prompt."""
        logging.info("Formatting retrieved context for final synthesis.")
        context_str = ""
        for i, item in enumerate(context):
            source_info = item.get('source', {})
            content = item.get('content', 'No content available.')
            
            # Create a clean, readable citation string
            citation = "N/A"
            source_type = source_info.get("type")
            if source_type == "graph_path":
                primary_src = source_info.get("primary_source", {})
                citation = f"[graph: {primary_src.get('type', 'UNKNOWN_REL')}]"
            elif "document_id" in source_info: # It's from Pinecone
                doc_id = source_info.get("document_id", "N/A")
                page = source_info.get("page_number", "N/A")
                citation = f"[doc: {doc_id}, page: {page}]"

            context_str += f"--- Context Source {i+1} ---\n"
            context_str += f"Source: {citation}\n"
            context_str += f"Content: {content}\n\n"
        
        return context_str

    def _synthesize_answer(self, query: str, context_str: str) -> str:
        """Uses an LLM call to synthesize the final answer from the context."""
        logging.info("Step 4: Synthesizing final answer from context.")
        prompt = prompts.SYNTHESIS_PROMPT.format(
            persona=self.persona,
            question=query,
            context_str=context_str
        )
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error during final answer synthesis: {e}")
            return "I apologize, but I encountered an error while trying to formulate a response."


    def run(self, query: str) -> str:
        """
        The main execution method for the agent.

        Args:
            query (str): The user's input query.

        Returns:
            str: The generated, cited answer.
        """
        if not self.llm:
            return "The AI model is not available. Please check the configuration."
        
        # 1. Analyze Intent
        intent = self._get_intent(query)
        
        # 2. Get Retrieval Strategy from PersonaRouter
        logging.info(f"Step 2: Getting retrieval plan for persona '{self.persona}'.")
        namespaces_to_search = self.router.get_retrieval_plan(self.persona)
        logging.info(f"Plan: Search namespaces {namespaces_to_search}")

        # 3. Dispatch to Tools (The "Second Brain")
        logging.info("Step 3: Dispatching tasks to specialist tools.")
        retrieved_context = []
        
        # 3a. Semantic Search (Pinecone)
        for namespace in namespaces_to_search:
            pinecone_results = tools.pinecone_search_tool(query=query, namespace=namespace)
            if pinecone_results:
                retrieved_context.extend(pinecone_results)
        
        # 3b. Graph Search (Neo4j) - if intent is appropriate
        if intent == "graph_query":
            cypher_query = self._generate_cypher(query)
            if cypher_query:
                graph_results = tools.neo4j_graph_tool(cypher_query=cypher_query)
                if graph_results:
                    retrieved_context.extend(graph_results)

        # 4. Check for Content and Synthesize
        if not retrieved_context:
            logging.warning("No context was retrieved from any data source.")
            return "Based on the provided documents and knowledge graph, I could not find any relevant information to answer this question."

        formatted_context = self._format_context_for_prompt(retrieved_context)
        
        final_answer = self._synthesize_answer(query, formatted_context)
        
        logging.info("Agent run completed.")
        return final_answer