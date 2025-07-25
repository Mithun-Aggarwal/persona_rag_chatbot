# src/common_utils.py

"""
Common utility functions shared across the application.
"""

from pathlib import Path

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.

    This is a robust way to reference files (like configs) from anywhere
    within the project, regardless of where the script is run from.
    """
    # We assume this file is in 'src/'. The project root is one level up.
    return Path(__file__).parent.parent.resolve()