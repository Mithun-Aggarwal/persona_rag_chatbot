# FILE: src/retrievers/pinecone.py

import logging
from typing import List
from sentence_transformers import SentenceTransformer
from src.models import ContextItem
import pinecone
import os

logger = logging.getLogger(__name__)

# Initialize Pinecone only once
def _init_pinecone():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    if not pinecone_api_key or not pinecone_env:
        raise RuntimeError("Pinecone environment not configured.")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Main vector search function
def vector_search(query: str, namespace: str, top_k: int = 10) -> List[ContextItem]:
    from src.models import Source  # Local import to avoid circular dependency

    _init_pinecone()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME not set in environment.")

    index = pinecone.Index(index_name)

    logger.info(f"🔎 Performing vector search on Pinecone index '{index_name}' in namespace '{namespace}'")

    model = SentenceTransformer("all-mpnet-base-v2")
    query_vector = model.encode(query).tolist()

    response = index.query(vector=query_vector, namespace=namespace, top_k=top_k, include_metadata=True)
    results = []
    for match in response.matches:
        metadata = match.metadata
        results.append(ContextItem(
            content=metadata.get("text", ""),
            score=match.score,
            source=Source(
                type="pinecone",
                document_id=metadata.get("doc_id"),
                page_numbers=[metadata.get("page")],
                chunk_id=metadata.get("chunk_id"),
                public_url=metadata.get("public_url")
            )
        ))

    return results
