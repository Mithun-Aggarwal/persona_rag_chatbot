#!/bin/bash
# FILE: clear_cache.sh
# This script finds and removes all __pycache__ directories
# and .pyc files from the project directory.

echo "--- Clearing Python Cache ---"

# Find all __pycache__ directories and remove them
find . -type d -name "__pycache__" -exec rm -r {} +

# Find any orphaned .pyc files and remove them
find . -type f -name "*.pyc" -delete

echo "✅ Python cache cleared successfully."