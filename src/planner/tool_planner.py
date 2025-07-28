# FILE: src/planner/tool_planner.py
# V2.0: Persona-Aware Tool Planner
import logging
import yaml
from pathlib import Path
from typing import List, Dict

from src.models import QueryMetadata, ToolPlanItem

logger = logging.getLogger(__name__)

# Base scores for tools based on query intent. A higher score is better.
# These tool names MUST match the names in `persona_tool_map.yml` and the tool router.
INTENT_TOOL_SCORES = {
    # If the user wants a specific fact...
    "specific_fact_lookup": {
        "query_knowledge_graph": 0.9,
        "retrieve_clinical_data": 0.7,
        "retrieve_general_text": 0.6,
        "retrieve_summary_data": 0.4,
    },
    # If the user wants a high-level summary...
    "simple_summary": {
        "retrieve_summary_data": 0.9,
        "retrieve_general_text": 0.8,
        "retrieve_clinical_data": 0.5,
        "query_knowledge_graph": 0.4,
    },
    # If the user wants to compare things...
    "comparative_analysis": {
        "retrieve_clinical_data": 0.8,
        "retrieve_summary_data": 0.8,
        "retrieve_general_text": 0.7,
        "query_knowledge_graph": 0.5,
    },
    # For general questions...
    "general_qa": {
        "retrieve_general_text": 0.9,
        "retrieve_summary_data": 0.7,
        "query_knowledge_graph": 0.6,
        "retrieve_clinical_data": 0.5,
    },
}
DEFAULT_INTENT_SCORE = 0.5
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class ToolPlanner:
    def __init__(self, coverage_threshold: float = 0.9):
        self.coverage_threshold = coverage_threshold
        self._load_persona_map()

    def _load_persona_map(self):
        """Loads the persona-to-tool mapping from the central YAML config."""
        map_file = PROJECT_ROOT / "config" / "persona_tool_map.yml"
        try:
            with open(map_file, 'r') as f:
                self.persona_map = yaml.safe_load(f)
            logger.info(f"Successfully loaded persona-tool map from '{map_file}'.")
        except Exception as e:
            logger.error(f"FATAL: Could not load or parse persona-tool map from '{map_file}': {e}", exc_info=True)
            self.persona_map = {}

    def plan(self, query_meta: QueryMetadata, persona: str) -> List[ToolPlanItem]:
        """
        Creates a ranked tool plan by combining query intent with user persona preferences.
        """
        logger.info(f"Planning tools for intent '{query_meta.intent}' and persona '{persona}'")
        
        # 1. Get the list of preferred tools and their weights for the given persona
        persona_key = persona.lower().replace(" ", "_")
        persona_prefs = self.persona_map.get(persona_key, self.persona_map.get("default", []))

        if not persona_prefs:
            logger.warning(f"No tool preferences found for persona '{persona_key}' or default. Returning empty plan.")
            return []
            
        persona_tool_weights: Dict[str, float] = {p["tool_name"]: p["weight"] for p in persona_prefs}
        
        # 2. Get intent-based scores for tools relevant to the current query intent
        intent_scores = INTENT_TOOL_SCORES.get(query_meta.intent, {})

        # 3. Calculate a final score for each tool by multiplying persona weight and intent score
        scored_tools = []
        for tool_name, persona_weight in persona_tool_weights.items():
            intent_score = intent_scores.get(tool_name, DEFAULT_INTENT_SCORE)
            
            # The final score reflects both the persona's general preference and the tool's suitability for the task
            final_score = persona_weight * intent_score
            scored_tools.append({"name": tool_name, "score": final_score})

        # 4. Sort tools by their final score in descending order
        scored_tools.sort(key=lambda x: x["score"], reverse=True)

        # 5. Build the final plan, adding tools until the cumulative coverage threshold is met
        final_plan: List[ToolPlanItem] = []
        total_coverage = 0.0
        for tool in scored_tools:
            # We treat the score as the estimated coverage for this planning step
            estimated_coverage = round(tool["score"], 2)

            # Do not add tools with negligible contribution
            if estimated_coverage <= 0.1:
                continue

            plan_item = ToolPlanItem(tool_name=tool["name"], estimated_coverage=estimated_coverage)
            final_plan.append(plan_item)
            
            total_coverage += estimated_coverage
            if total_coverage >= self.coverage_threshold:
                logger.info(f"Coverage threshold of {self.coverage_threshold} met. Finalizing plan.")
                break
        
        logger.info(f"Generated tool plan: {[t.model_dump_json(indent=2) for t in final_plan]}")
        return final_plan