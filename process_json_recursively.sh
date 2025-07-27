#!/bin/bash

# ==============================================================================
# SCRIPT: process_files_recursively_and_combine.sh (v2 - JSONL Support)
#
# DESCRIPTION:
#   This script recursively finds all .json AND .jsonl files in a specified
#   directory and its subdirectories. For each file, it does two things:
#
#   1. Creates an individual .txt file containing the first N lines.
#   2. Appends the content of that new file into a single, combined .txt file.
#
#   This is ideal for preparing data for LLM analysis, allowing for both
#   individual file inspection and combined contextual analysis.
#
# USAGE:
#   ./process_files_recursively_and_combine.sh [path/to/your/folder]
# ==============================================================================

# --- Configuration ---
# The number of lines (rows) to keep from the top of each file.
LINES_TO_KEEP=1000

# The directory to search. Defaults to the current directory "."
TARGET_DIR="${1:-.}"

# The name of the subdirectory where all new .txt files will be saved.
OUTPUT_DIR="processed_txt_files"

# The name for the single file that will contain all the combined text.
# The leading underscore helps it sort to the top of the file list.
COMBINED_FILENAME="_all_combined_content.txt"

# --- Pre-flight Checks ---

# Resolve the absolute path of the target directory for robust path manipulation.
TARGET_DIR=$(realpath "$TARGET_DIR")

# Check if the target directory actually exists.
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' not found."
  exit 1
fi

# Create the output directory. The "-p" flag prevents errors if it already exists.
mkdir -p "${TARGET_DIR}/${OUTPUT_DIR}"

# Define the full path for the combined output file.
COMBINED_OUTPUT_PATH="${TARGET_DIR}/${OUTPUT_DIR}/${COMBINED_FILENAME}"

# Initialize the combined file by overwriting it with a header.
# This ensures it's clean before the script starts appending data.
echo "--- Combined output of all processed .json and .jsonl files. Generated on $(date) ---" > "$COMBINED_OUTPUT_PATH"

echo "Recursively searching for .json and .jsonl files in: $TARGET_DIR"
echo "Processed files will be saved in: ${TARGET_DIR}/${OUTPUT_DIR}"
echo "--------------------------------------------------"

# --- Main Processing Loop ---

file_count=0
# --- UPDATED FIND COMMAND: Uses `-o` for OR logic to find both file types ---
find "$TARGET_DIR" -type f \( -name "*.json" -o -name "*.jsonl" \) -not -path "*/${OUTPUT_DIR}/*" | while IFS= read -r source_file; do
  
  ((file_count++))
  echo "Processing: $source_file"

  # --- Create the individual .txt file ---
  relative_path="${source_file#$TARGET_DIR/}"
  # Sanitize the name by replacing extension and slashes
  sanitized_name="${relative_path//\//_}"
  sanitized_name="${sanitized_name%.json}"
  sanitized_name="${sanitized_name%.jsonl}"
  
  timestamp=$(date +%Y%m%d_%H%M%S_%N) # Added nanoseconds (%N) for better uniqueness
  new_txt_filename="${sanitized_name}_${timestamp}.txt"
  output_path="${TARGET_DIR}/${OUTPUT_DIR}/${new_txt_filename}"

  # Use 'head' to take the first N lines and create the new individual .txt file.
  head -n "$LINES_TO_KEEP" "$source_file" > "$output_path"
  echo "  -> Saved individual file: $new_txt_filename"

  # --- Append to the combined file ---
  # Add a clear separator to the combined file to distinguish between sources.
  echo -e "\n\n# ======================================================" >> "$COMBINED_OUTPUT_PATH"
  echo "# START OF CONTENT FROM: ${relative_path}" >> "$COMBINED_OUTPUT_PATH"
  echo "# ======================================================\n" >> "$COMBINED_OUTPUT_PATH"
  
  # Append the content of the newly created individual file to the combined file.
  cat "$output_path" >> "$COMBINED_OUTPUT_PATH"
  
  echo "  -> Appended content to: $COMBINED_FILENAME"

done

# --- Final Report ---
echo "--------------------------------------------------"
if [ "$file_count" -eq 0 ]; then
  echo "No .json or .jsonl files were found in '$TARGET_DIR' or its subdirectories."
else
  echo "Processing complete. Total files processed: $file_count."
  echo "All content has also been aggregated into the master file:"
  echo "  -> ${COMBINED_OUTPUT_PATH}"
fi