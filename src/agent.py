# src/agent.py

import logging
from typing import List
import google.generativeai as genai

from src import tools, prompts, retrievers
from src.models import ContextItem
from src.routing.persona_router import PersonaRouter

logger = logging.getLogger(__name__)

class MainAgent:
    def __init__(self, persona: str):
        self.persona = persona
        self.router = PersonaRouter()
        self.llm = genai.GenerativeModel('gemini-1.5-flash-latest')

    def _generate_cypher(self, query: str) -> str | None:
        logger.info("Attempting to generate Cypher query...")
        try:
            # This now correctly calls the new function in tools.py
            live_schema = tools.get_neo4j_schema()
            if "Error:" in live_schema:
                logger.error(f"Could not generate Cypher, failed to get schema: {live_schema}")
                return None
        except Exception as e:
            logger.error(f"Could not generate Cypher, failed to get schema: {e}")
            return None

        prompt = prompts.CYPHER_GENERATION_PROMPT.format(
            schema=live_schema,
            question=query
        )
        try:
            response = self.llm.generate_content(prompt)
            cypher_query = response.text.strip().replace("```cypher", "").replace("```", "") # Clean up markdown

            if "NONE" in cypher_query.upper() or "MATCH" not in cypher_query.upper():
                logger.warning("LLM determined question is not suitable for graph query.")
                return None

            logger.info(f"Successfully generated Cypher: {cypher_query}")
            return cypher_query
        except Exception as e:
            logger.error(f"Error during Cypher generation: {e}")
            return None

    def _format_context_with_citations(self, context: List[ContextItem]) -> str:
        logger.info("Formatting context and creating citation markers.")
        context_str = ""
        source_map = {}
        ref_counter = 1

        for item in context:
            source = item.source
            if source.type in ["graph_path", "graph_record"]:
                # Use a cleaner key for graph results
                source_key = f"Graph DB Record (Query: '{source.query[:60]}...')"
            elif source.document_id and source.page_numbers:
                page_str = ", ".join(map(str, sorted(list(set(source.page_numbers)))))
                source_key = f"Document: {source.document_id}, Page(s): {page_str}"
            elif source.document_id:
                source_key = f"Document: {source.document_id}"
            else:
                continue # Skip items with no identifiable source

            if source_key not in source_map:
                source_map[source_key] = f"[{ref_counter}]"
                ref_counter += 1

            citation_marker = source_map[source_key]

            context_str += f"--- Context Source ---\n"
            context_str += f"Source Citation: {citation_marker}\n"
            context_str += f"Content: {item.content}\n\n"

        # Only add reference list if there are sources
        if source_map:
            reference_list = "\n".join([f"{num} {key}" for key, num in source_map.items()])
            context_str += f"\n--- Available References ---\n{reference_list}\n"
        return context_str

    def run(self, query: str) -> str:
        logger.info(f"\U0001F7E2 Agent starting run for persona '{self.persona}' with query: '{query}'")

        retrieval_plan = self.router.get_retrieval_plan(self.persona)
        retrieved_context: List[ContextItem] = []

        # Retrieve from vector stores first
        unique_content = set()
        for config in retrieval_plan.namespaces:
            logger.info(f"\U0001F50D Executing vector search on namespace: {config.namespace} with top_k: {config.top_k}")
            # This now correctly calls the function in retrievers.py
            pinecone_results = retrievers.vector_search(query=query, namespace=config.namespace, top_k=config.top_k)
            for res in pinecone_results:
                if res.content not in unique_content:
                    retrieved_context.append(res)
                    unique_content.add(res.content)
        
        # Then, attempt to retrieve from graph store
        cypher_query = self._generate_cypher(query)
        if cypher_query:
            logger.info(f"\U0001F578️ Executing graph search with Cypher: {cypher_query}")
            # This now correctly calls the function in retrievers.py
            graph_results = retrievers.graph_search(cypher_query=cypher_query)
            retrieved_context.extend(graph_results)

        if not retrieved_context:
            logger.warning("No information found from any retriever.")
            return "Based on the information available to me, I could not find a sufficient answer to your question."

        logger.info(f"Retrieved {len(retrieved_context)} total context items. Synthesizing answer.")
        formatted_context = self._format_context_with_citations(retrieved_context)
        final_prompt = prompts.SYNTHESIS_PROMPT.format(question=query, context_str=formatted_context)

        try:
            final_answer = self.llm.generate_content(final_prompt).text
        except Exception as e:
            logger.error(f"Error during final synthesis: {e}")
            return "I apologize, but I encountered an error while formulating a response."

        logger.info("Agent run completed successfully.")
        return final_answer