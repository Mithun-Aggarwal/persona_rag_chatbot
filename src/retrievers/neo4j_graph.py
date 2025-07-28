# FILE: src/retrievers/neo4j_graph.py
# Phase 2.3: Neo4jGraphAgent — safe Cypher runner with relationship path output

import logging
from typing import List
from neo4j import GraphDatabase, exceptions

from src.models import ToolResult, QueryMetadata
from src.tools import get_neo4j_driver, serialize_neo4j_path

logger = logging.getLogger(__name__)


def run_neo4j_search(query: str, query_meta: QueryMetadata) -> ToolResult:
    """
    Executes Cypher query safely. First dry-runs with RETURN 1 to validate.
    If valid, executes and serializes relationship paths.
    """
    driver = get_neo4j_driver()
    if not driver:
        return ToolResult(tool_name="neo4j", success=False, content="Neo4j driver not available.")

    cypher = generate_cypher(query, query_meta)
    if not cypher:
        return ToolResult(tool_name="neo4j", success=False, content="Query not graph-suitable or no Cypher generated.")

    if not _is_valid_cypher(driver, cypher):
        return ToolResult(tool_name="neo4j", success=False, content="Generated Cypher query failed validation.")

    try:
        with driver.session() as session:
            records = session.run(cypher)
            paths = []

            for rec in records:
                p = rec.get("p")
                if p:
                    paths.append(serialize_neo4j_path(p))
                else:
                    paths.append(str(rec))

            if not paths:
                return ToolResult(tool_name="neo4j", success=False, content="No matching graph paths found.")

            return ToolResult(tool_name="neo4j", success=True, content="\n---\n".join(paths))
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}", exc_info=True)
        return ToolResult(tool_name="neo4j", success=False, content=f"Error during execution: {e}")


def generate_cypher(query: str, query_meta: QueryMetadata) -> str:
    """Placeholder Cypher generation. Later replaced with LLM-based logic."""
    if not query_meta.question_is_graph_suitable:
        return ""
    return f"MATCH p=(d:Drug)-[r]->(x) WHERE d.name CONTAINS '{query_meta.keywords[0]}' RETURN p LIMIT 5"


def _is_valid_cypher(driver, cypher: str) -> bool:
    try:
        with driver.session() as session:
            test_query = f"CALL {{ {cypher} }} RETURN 1"
            session.run(test_query)
            return True
    except exceptions.CypherSyntaxError as e:
        logger.warning(f"Dry-run validation failed: {e}")
        return False
