from pydantic import BaseModel, Field
from typing import List, Optional, Literal


QueryIntent = Literal[
    "specific_fact_lookup",
    "simple_summary",
    "comparative_analysis",
    "general_qa",
    "unknown"
]


class QueryMetadata(BaseModel):
    intent: QueryIntent
    keywords: List[str]
    question_is_graph_suitable: bool
    themes: Optional[List[str]] = Field(default_factory=list, description="High-level themes for metadata filtering.")


class ToolPlanItem(BaseModel):
    tool_name: str
    estimated_coverage: float


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    content: str
    estimated_coverage: float = 0.0


class ContextItem(BaseModel):
    content: str
    source: Optional[dict] = None


class Source(BaseModel):
    type: str
    document_id: Optional[str] = None
    page_numbers: Optional[List[int]] = None
    source_url: Optional[str] = None
    retrieval_score: Optional[float] = None
    query: Optional[str] = None

class NamespaceConfig(BaseModel):
    namespace: str
    weight: float
    top_k: int

class RetrievalPlan(BaseModel):
    namespaces: List[NamespaceConfig] = []