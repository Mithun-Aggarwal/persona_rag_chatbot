"""PineconeRetriever v2 — vector search with persona filter + optional re‑rank.

Drop this in `src/retrievers/pinecone.py` and import from `ToolRouter` once you
replace the earlier stub.  The retriever assumes text chunks are already
embedded and stored under the configured index name.

Config via ENV
-------------
PINECONE_API_KEY       your Pinecone key (required)
PINECONE_ENVIRONMENT   environment string (default "gcp-starter")
PINECONE_INDEX_NAME    name of the vector index (required)
RE_RANK                "true" to enable Cohere re‑rank (optional)

Usage
-----
>>> from retrievers.pinecone import PineconeRetriever
>>> retriever = PineconeRetriever()
>>> resp = retriever.search("belantamab mafodotin PBAC outcome", top_k=5, persona="clinical")
>>> print(resp[0].text, resp[0].score)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import retrivers.retrievers as retrievers  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pinecone-client not installed. `pip install pinecone-client`") from exc

# Optional Cohere re-ranker
try:
    import cohere  # type: ignore
except ImportError:
    cohere = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("sentence-transformers required. `pip install sentence-transformers`") from exc


@dataclass
class RetrievalChunk:
    id: str
    text: str
    score: float
    metadata: dict


class PineconeRetriever:
    """Encapsulates vector search with persona filter + (optional) re‑rank."""

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        namespace: Optional[str] = None,
        top_k_default: int = 10,
    ):
        self.namespace = namespace
        self.top_k_default = top_k_default
        self.re_rank_enabled = os.getenv("RE_RANK", "false").lower() == "true" and cohere is not None

        # --- init Pinecone ---
        retrievers.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
        )
        index_name = os.environ["PINECONE_INDEX_NAME"]
        self.index = retrievers.Index(index_name)

        # --- embedder ---
        self.embedder = SentenceTransformer(model_name)

        # --- Cohere client for re‑rank (optional) ---
        self._co = None
        if self.re_rank_enabled:
            self._co = cohere.Client(os.environ.get("COHERE_API_KEY"))

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        persona: str | None = None,
    ) -> List[RetrievalChunk]:
        """Return ranked list of RetrievalChunk."""

        vec = self.embedder.encode(query).tolist()
        top_k = top_k or self.top_k_default

        # Build Pinecone filter
        filter_dict = {"persona_scores." + persona: {"$gte": 0.7}} if persona else None

        # Query
        pc_res = self.index.query(
            vector=vec,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
            include_values=False,
            filter=filter_dict,
        )

        chunks = [
            RetrievalChunk(
                id=match["id"],
                text=match["metadata"].get("text", ""),
                score=match["score"],
                metadata=match["metadata"],
            )
            for match in pc_res["matches"]
        ]

        # Optional re‑rank with Cohere
        if self.re_rank_enabled and chunks:
            chunks = self._rerank(query, chunks)
        return chunks

    # ------------------------------------------------------------------
    def _rerank(self, query: str, chunks: List[RetrievalChunk]) -> List[RetrievalChunk]:
        assert self._co, "Cohere client missing"
        docs = [c.text for c in chunks]
        rerank_res = self._co.rerank(query=query, documents=docs)
        # Cohere returns best‑to‑worst indices
        new_order = [chunks[i] for i in rerank_res["reranked_indices"]]
        # Update scores to Cohere relevance 0‑1
        for c, score in zip(new_order, rerank_res["relevance_scores"]):
            c.score = score
        return new_order


# ---------------------------------------------------------------------------
# CLI quick test (embedding + search latency)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import json
    retriever = PineconeRetriever()
    q = "belantamab mafodotin pbac positive recommendation"
    t0 = time.time()
    out = retriever.search(q, top_k=3, persona="clinical")
    print(json.dumps([c.__dict__ for c in out], indent=2))
    print(f"Latency: {time.time() - t0:.2f}s")
