# FILE: src/middleware/logging.py
# Phase 3.2: LoggingMiddleware — structured trace logs per query execution

import json
import time
import logging
from pathlib import Path
from typing import List
from datetime import datetime

from src.models import ToolResult, ToolPlanItem, QueryMetadata

logger = logging.getLogger(__name__)

LOG_PATH = Path("trace_logs.jsonl")  # Can be adjusted per environment


def log_trace(
    query: str,
    query_meta: QueryMetadata,
    tool_plan: List[ToolPlanItem],
    tool_results: List[ToolResult],
    total_latency_sec: float
):
    """
    Writes a structured JSON line capturing the full agent loop.
    """
    trace_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "intent": query_meta.intent,
        "graph_suitable": query_meta.question_is_graph_suitable,
        "keywords": query_meta.keywords,
        "tool_plan": [t.model_dump() for t in tool_plan],
        "tool_results": [r.model_dump() for r in tool_results],
        "total_latency_sec": round(total_latency_sec, 3)
    }

    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_record) + "\n")
        logger.info(f"Trace logged to {LOG_PATH.resolve()}")
    except Exception as e:
        logger.error(f"Failed to write trace log: {e}", exc_info=True)


# Optional context manager for timing
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start