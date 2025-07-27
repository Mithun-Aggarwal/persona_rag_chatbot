# src/agent.py (V-Final)

import streamlit as st
import google.generativeai as genai
import logging
from typing import List, Dict, Any

from src import prompts, tools
from src.routing.persona_router import PersonaRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

class MainAgent:
    def __init__(self, persona: str):
        self.persona = persona
        self.router = PersonaRouter()
        self.llm = genai.GenerativeModel('gemini-1.5-flash-latest')

    def _generate_cypher(self, query: str) -> str | None:
        logger.info("Attempting to generate Cypher query using dynamic schema...")
        
        live_schema = tools.get_neo4j_schema()
        
        prompt = prompts.CYPHER_GENERATION_PROMPT.format(
            schema=live_schema,
            question=query
        )
        try:
            response = self.llm.generate_content(prompt)
            # --- THE FIX: Clean the markdown fences from the LLM's output ---
            cypher_query = response.text.strip().replace("```cypher", "").replace("```", "").strip()
            
            if "NONE" in cypher_query or "MATCH" not in cypher_query.upper():
                logger.warning("LLM determined question is not suitable for graph query based on live schema.")
                return None
                
            logger.info(f"Successfully generated Cypher using live schema: {cypher_query}")
            return cypher_query
        except Exception as e:
            logger.error(f"Error during Cypher generation: {e}")
            return None

    def _format_context_with_citations(self, context: List[Dict[str, Any]]) -> str:
        logger.info("Formatting context and creating citation markers.")
        context_str = ""
        # Create a mapping from a unique source description to a reference number
        source_map = {}
        ref_counter = 1
        
        for item in context:
            source_info = item.get('source', {})
            content = item.get('content', 'No content available.')
            doc_id = source_info.get("document_id", "N/A")
            
            # Create a unique, readable source string
            pages = source_info.get("page_numbers")
            source_key = ""
            if isinstance(pages, list) and pages:
                page_str = ", ".join(map(str, sorted(list(set(pages)))))
                source_key = f"doc: {doc_id}, page: {page_str}"
            elif source_info.get("type") == "graph_path":
                source_key = f"Graph Query Result for '{item['content'][:50]}...'"
            elif doc_id != "N/A":
                source_key = f"doc: {doc_id}"
            else:
                continue # Skip context with no identifiable source
                
            if source_key not in source_map:
                source_map[source_key] = f"[{ref_counter}]"
                ref_counter += 1
            
            citation_marker = source_map[source_key]

            context_str += f"--- Context Source ---\n"
            context_str += f"Source Citation: {citation_marker}\n"
            context_str += f"Content: {content}\n\n"
        
        # Also create the reference list to append to the prompt
        reference_list = "\n".join([f"{num} {key}" for key, num in source_map.items()])
        context_str += f"\n--- Available References ---\n{reference_list}\n"
        return context_str
    
    def run(self, query: str) -> str:
        if not self.llm: return "The AI model is not available."

        # Step 1: Simple, Robust Retrieval
        logger.info("Starting robust retrieval process.")
        retrieved_context = []
        namespaces = self.router.get_retrieval_plan(self.persona)
        
        # 1a. Vector Search (on original query only)
        unique_content = set()
        for namespace in namespaces:
            pinecone_results = tools.pinecone_search_tool(query=query, namespace=namespace)
            for res in pinecone_results:
                content = res.get('content')
                if content and content not in unique_content:
                    retrieved_context.append(res)
                    unique_content.add(content)

        # 1b. Graph Search (on original query only)
        cypher_query = self._generate_cypher(query)
        if cypher_query:
            graph_results = tools.neo4j_graph_tool(cypher_query=cypher_query)
            if graph_results: retrieved_context.extend(graph_results)
            
        if not retrieved_context:
            return "Based on the provided documents, I could not find sufficient information to answer this question."

        # Step 2: Format Context and Synthesize in a Single, Powerful Call
        # The prompt now instructs the LLM to do all the work, which is more reliable.
        formatted_context = self._format_context_with_citations(retrieved_context)
        final_prompt = prompts.SYNTHESIS_PROMPT.format(question=query, context_str=formatted_context)
        
        try:
            logger.info("Synthesizing final answer using single-pass method.")
            final_answer = self.llm.generate_content(final_prompt).text
        except Exception as e:
            logger.error(f"Error during final synthesis: {e}")
            return "I apologize, but I encountered a critical error while formulating a response."

        logger.info("Agent run completed successfully.")
        return final_answer