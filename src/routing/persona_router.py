# FILE: src/routing/persona_router.py
# V1.0: Implements the PersonaRouter class for intelligent, config-driven query planning.

import logging
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path to allow imports from src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.utils import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

class PersonaRouter:
    """
    Loads a persona-to-namespace map and provides query plans for a RAG system.
    
    This class is the core of the persona-aware retrieval strategy. It translates a
    high-level persona name into a concrete, weighted list of Pinecone namespaces
    that should be queried to find the most relevant information for that user.
    """
    
    def __init__(self, map_file_path: Path = None):
        """
        Initializes the router by loading the persona-to-namespace map.
        
        Args:
            map_file_path: Optional path to the map file. If None, it defaults to
                           `config/persona_namespace_map.yml` in the project root.
        """
        if map_file_path is None:
            self.map_file_path = get_project_root() / "config" / "persona_namespace_map.yml"
        else:
            self.map_file_path = map_file_path
            
        self.persona_map = self._load_map()

    def _load_map(self) -> Dict[str, Any]:
        """
        Loads and parses the YAML file that defines the routing strategy.
        
        Returns:
            A dictionary representing the parsed YAML, or an empty dict on failure.
        """
        try:
            logging.info(f"Loading persona routing map from: {self.map_file_path}")
            with open(self.map_file_path, "r", encoding="utf-8") as f:
                persona_map = yaml.safe_load(f)
            if not persona_map or not isinstance(persona_map, dict):
                logging.error("Persona map is empty or not a valid dictionary. Router will be ineffective.")
                return {}
            logging.info("Successfully loaded persona routing map.")
            return persona_map
        except FileNotFoundError:
            logging.critical(f"FATAL: Persona map file not found at {self.map_file_path}. The router cannot function.")
            return {}
        except yaml.YAMLError as e:
            logging.critical(f"FATAL: Error parsing YAML from {self.map_file_path}. Error: {e}")
            return {}

    def get_query_plan(self, persona: str) -> List[Dict[str, Any]]:
        """
        Gets the weighted list of namespaces for a given persona.
        
        If the specified persona is not found in the map, it gracefully falls
        back to the 'default' persona's query plan. If 'default' is also missing,
        it returns an empty list.

        Args:
            persona: The name of the persona (e.g., 'clinical_analyst').

        Returns:
            A list of dictionaries, where each dictionary contains 'namespace' and 'weight'.
            Example: [{'namespace': 'pbac-clinical', 'weight': 1.0}, ...]
        """
        if persona in self.persona_map:
            logging.info(f"Found specific query plan for persona: '{persona}'")
            return self.persona_map[persona]
        else:
            logging.warning(f"Persona '{persona}' not found. Falling back to 'default' query plan.")
            return self.persona_map.get('default', [])

# --- Example Usage (for testing purposes) ---
if __name__ == "__main__":
    print("--- PersonaRouter Test Harness ---")
    
    # Initialize the router. It will automatically find the config file.
    router = PersonaRouter()
    
    print("\n--- Testing a defined persona: 'clinical_analyst' ---")
    clinical_plan = router.get_query_plan("clinical_analyst")
    if clinical_plan:
        for step in clinical_plan:
            print(f"  -> Query namespace '{step['namespace']}' with weight {step['weight']}")
    else:
        print("  -> No plan found.")

    print("\n--- Testing a defined persona: 'health_economist' ---")
    he_plan = router.get_query_plan("health_economist")
    if he_plan:
        for step in he_plan:
            print(f"  -> Query namespace '{step['namespace']}' with weight {step['weight']}")
    else:
        print("  -> No plan found.")
        
    print("\n--- Testing an UNDEFINED persona: 'data_scientist' (should fall back to default) ---")
    unknown_plan = router.get_query_plan("data_scientist")
    if unknown_plan:
        for step in unknown_plan:
            print(f"  -> Query namespace '{step['namespace']}' with weight {step['weight']}")
    else:
        print("  -> No plan found (and no default plan exists).")