# FILE: src/common/utils.py
import yaml
from pathlib import Path

def load_config(file_path: str) -> dict:
    with open(Path(file_path), 'r') as f:
        return yaml.safe_load(f)