# src/routing/persona_router.py

import logging
import yaml
from pathlib import Path
from typing import List

from src.common_utils import get_project_root
from src.models import RetrievalPlan, NamespaceConfig # REFACTORED: Use Pydantic models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# REFACTORED: A constant to scale retrieval depth by weight.
# A weight of 1.0 will result in top_k=15, 0.5 will result in top_k=7.
BASE_TOP_K = 15

class PersonaRouter:
    """
    Loads a persona-to-namespace map and creates a structured retrieval plan.
    It translates a persona into a list of weighted, configured namespaces to query.
    """
    def __init__(self, map_file_path: Path = None):
        """
        Initializes the router by loading the persona-to-namespace map.
        """
        if map_file_path is None:
            self.map_file_path = get_project_root() / "config" / "persona_namespace_map.yml"
        else:
            self.map_file_path = map_file_path

        try:
            with open(self.map_file_path, 'r') as f:
                self.persona_map = yaml.safe_load(f)
            logger.info(f"Successfully loaded persona map from {self.map_file_path}")
        except FileNotFoundError:
            logger.error(f"FATAL: Persona map file not found at {self.map_file_path}")
            # REFACTORED: Raise an exception instead of calling st.error
            raise
        except Exception as e:
            logger.error(f"FATAL: Error loading or parsing persona map file: {e}")
            self.persona_map = {}

    def get_retrieval_plan(self, persona: str) -> RetrievalPlan:
        """
        Gets the structured retrieval plan for a given persona.

        Args:
            persona: The name of the persona (e.g., 'Clinical Analyst').

        Returns:
            A RetrievalPlan object containing a list of configured namespaces.
        """
        normalized_persona = persona.lower().replace(" ", "_")
        persona_configs = self.persona_map.get(normalized_persona, self.persona_map.get('default', []))

        if not persona_configs:
            logger.warning(f"No plan found for persona '{normalized_persona}' or default. Returning empty plan.")
            return RetrievalPlan()

        # REFACTORED: Create NamespaceConfig models, now using the 'weight'
        namespace_configs = []
        for item in persona_configs:
            weight = item.get('weight', 1.0)
            # Dynamically calculate top_k based on weight
            top_k = int(max(3, BASE_TOP_K * weight))
            
            config = NamespaceConfig(
                namespace=item['namespace'],
                weight=weight,
                top_k=top_k
            )
            namespace_configs.append(config)
        
        plan = RetrievalPlan(namespaces=namespace_configs)
        logger.info(f"Retrieval plan for persona '{persona}': {plan.model_dump_json(indent=2)}")
        return plan