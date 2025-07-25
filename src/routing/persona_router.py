# src/routing/persona_router.py

# V1.1: Implements the PersonaRouter class for intelligent, config-driven query planning.
# This version is updated to be self-contained and parse a weighted namespace map.

import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class PersonaRouter:
    """
    Loads a persona-to-namespace map and provides query plans for a RAG system.

    This class translates a high-level persona name into a concrete list of
    Pinecone namespaces that should be queried to find the most relevant
    information for that user. It reads this mapping from a YAML config file.
    """
    def __init__(self, map_file_path: Path = None):
        """
        Initializes the router by loading the persona-to-namespace map.

        Args:
            map_file_path: Optional path to the map file. If None, it defaults to
                           'config/persona_namespace_map.yml' in the project root.
        """
        if map_file_path is None:
            # Robustly find the project root relative to this file's location
            # (src/routing/persona_router.py -> go up 2 levels to project root)
            project_root = Path(__file__).resolve().parents[2]
            self.map_file_path = project_root / "config" / "persona_namespace_map.yml"
        else:
            self.map_file_path = map_file_path

        try:
            with open(self.map_file_path, 'r') as f:
                self.persona_map = yaml.safe_load(f)
            logging.info(f"Successfully loaded persona map from {self.map_file_path}")
        except FileNotFoundError:
            logging.error(f"FATAL: Persona map file not found at {self.map_file_path}")
            st.error(f"Persona config file not found. Please ensure `config/persona_namespace_map.yml` exists.")
            # In a real app, you might want to raise the exception or handle it differently
            self.persona_map = {}
        except Exception as e:
            logging.error(f"FATAL: Error loading or parsing persona map file: {e}")
            self.persona_map = {}

    def get_retrieval_plan(self, persona: str) -> List[str]:
        """
        Gets the list of namespaces for a given persona.

        Args:
            persona (str): The name of the persona (e.g., 'clinical_analyst').

        Returns:
            A list of namespace names (e.g., ['pbac-clinical', 'pbac-kg']).
        """
        # Normalize the persona string to match YAML keys (e.g., "Clinical Analyst" -> "clinical_analyst")
        normalized_persona = persona.lower().replace(" ", "_")

        persona_plan = self.persona_map.get(normalized_persona, self.persona_map.get('default', []))
        
        if not persona_plan:
            logging.warning(f"No plan found for persona '{normalized_persona}' or default. Returning empty list.")
            return []

        # Extract just the 'namespace' value from each dictionary in the list
        namespaces = [item['namespace'] for item in persona_plan if 'namespace' in item]
        
        logging.info(f"Retrieval plan for persona '{persona}': Query namespaces {namespaces}")
        return namespaces