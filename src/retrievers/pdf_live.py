# FILE: src/retrievers/pdf_live.py
# Phase 2.4: PDFLiveRetriever — stream, extract, and embed up to 3 PDFs or 100 pages total

import logging
import tempfile
import requests
from typing import List

from src.models import ToolResult, QueryMetadata
from src.tools import extract_pdf_text, get_google_ai_client, get_pinecone_index, stream_pdf_metadata

logger = logging.getLogger(__name__)

MAX_PDFS = 3
MAX_TOTAL_PAGES = 100


def run_pdf_retriever(query: str, query_meta: QueryMetadata) -> ToolResult:
    """
    Loads and processes up to 3 PDFs (or 100 pages total) from public URLs.
    Uses Google Embed API to get semantic representation and return top chunks.
    """
    try:
        pdf_links = stream_pdf_metadata(query_meta.keywords, max_docs=MAX_PDFS)
        if not pdf_links:
            return ToolResult(tool_name="pdf", success=False, content="No public PDFs matched keywords.")

        embedding_client = get_google_ai_client()
        pinecone_index = get_pinecone_index()
        query_embedding = embedding_client.embed_content(query=query, model="models/embedding-001", task_type="retrieval_query")["embedding"]

        total_pages, all_chunks = 0, []
        for url in pdf_links:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                r = requests.get(url, timeout=20)
                tmp.write(r.content)
                tmp.flush()

                chunks = extract_pdf_text(tmp.name)
                if not chunks: continue

                all_chunks.extend(chunks)
                total_pages += len(chunks)
                if total_pages >= MAX_TOTAL_PAGES:
                    break

        if not all_chunks:
            return ToolResult(tool_name="pdf", success=False, content="Could not extract any text from public PDFs.")

        # Embed all chunks and score against query
        embedded_chunks = embedding_client.embed_contents(
            model="models/embedding-001",
            contents=[c["text"] for c in all_chunks],
            task_type="retrieval_document"
        )["embeddings"]

        scored = [
            (chunk, _cosine_similarity(e, query_embedding))
            for chunk, e in zip(all_chunks, embedded_chunks)
        ]
        top_matches = sorted(scored, key=lambda x: -x[1])[:5]

        snippet = "\n---\n".join(f"• {c['text'][:500]}..." for c, _ in top_matches)
        return ToolResult(tool_name="pdf", success=True, content=snippet)

    except Exception as e:
        logger.error(f"PDFLiveRetriever failed: {e}", exc_info=True)
        return ToolResult(tool_name="pdf", success=False, content=f"Error: {e}")


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot / (norm1 * norm2 + 1e-8)