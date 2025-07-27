#!/bin/bash

# ==============================================================================
# SCRIPT: pack_project_for_llm_v5.sh
#
# DESCRIPTION:
#   A robust project packager that creates a precise, four-file context
#   package for LLMs. It is location-independent, 100% respects .gitignore,
#   and is resilient to discrepancies between the Git index and the filesystem.
#
# v5 IMPROVEMENTS:
#   - Adds a check '[ -f ... ]' to verify a file physically exists before
#     attempting to 'cat' it. This prevents "No such file or directory" errors
#     if a file is deleted locally but still tracked by Git.
# ==============================================================================

# --- Helper Function to Pack Files by Category ---
# Arguments: 1. Git Root Dir, 2. Regex pattern, 3. Output Filename, 4. Description
pack_files_by_category() {
    local git_root=$1
    local pattern=$2
    local output_filename=$3
    local description=$4
    local full_output_path="${git_root}/${output_filename}"
    local file_count=0
    local skipped_count=0

    echo "📦 Packing ${description}..."

    while IFS= read -r file_path; do
        local full_source_path="${git_root}/${file_path}"

        # --- NEW ROBUSTNESS CHECK ---
        # Verify the file actually exists on the filesystem before proceeding.
        # This handles cases where a file is deleted locally but still in the Git index.
        if [ ! -f "$full_source_path" ]; then
            ((skipped_count++))
            continue # Skip to the next file in the loop
        fi
        
        # Initialize file only when the first *valid* match is found
        if [ "$file_count" -eq 0 ]; then
            echo "========================================================================" > "$full_output_path"
            echo "  Combined ${description} from Git Repository: ${git_root}" >> "$full_output_path"
            echo "  Generated on: $(date)" >> "$full_output_path"
            echo "========================================================================" >> "$full_output_path"
        fi
        
        ((file_count++))
        echo "   -> Adding: $file_path"
        
        echo -e "\n\n" >> "$full_output_path"
        echo "########################################################################" >> "$full_output_path"
        echo "### FILE: ${file_path}" >> "$full_output_path"
        echo "########################################################################" >> "$full_output_path"
        echo "" >> "$full_output_path"
        cat "$full_source_path" >> "$full_output_path"
    done < <(git -C "$git_root" ls-files | grep -E "$pattern")

    if [ "$skipped_count" -gt 0 ]; then
      echo "   🟡 Skipped $skipped_count file(s) that are tracked by Git but do not exist on disk."
      echo "      (You may want to run 'git status' and 'git add .' to sync your repository)"
    fi

    if [ "$file_count" -eq 0 ]; then
        echo "   🟡 No tracked and existing ${description} found."
    else
        echo "   ✅ Combined $file_count files into ${output_filename}"
    fi
}

# --- Helper Function to Generate a clean, git-aware file structure list ---
generate_file_structure() {
    local git_root=$1
    local output_filename=$2
    local full_output_path="${git_root}/${output_filename}"
    
    echo "🌳 Generating project file structure (respecting .gitignore)..."
    echo "========================================================================" > "$full_output_path"
    echo "  Project File Structure for: ${git_root}" >> "$full_output_path"
    echo "  (This list is from 'git ls-files' and is 100% accurate to the Git index)" >> "$full_output_path"
    echo "========================================================================" >> "$full_output_path"
    echo "" >> "$full_output_path"
    git -C "$git_root" ls-files >> "$full_output_path"
    echo "   ✅ Project structure saved to ${output_filename}"
}

# --- Main Execution Block ---
main() {
    echo "🚀 Starting LLM Project Packager v5..."
    
    local git_root
    git_root=$(git rev-parse --show-toplevel 2>/dev/null)
    if [ -z "$git_root" ]; then
      echo "❌ Error: This script must be run from within a Git repository."
      exit 1
    fi
    echo "✅ Project root identified: $git_root"
    echo "--------------------------------------------------"

    local python_output_file="_combined_python_code.txt"
    local config_output_file="_combined_configs_and_prompts.txt"
    local shell_output_file="_combined_shell_scripts.txt"
    local structure_output_file="_project_structure.txt"

    generate_file_structure "$git_root" "$structure_output_file"
    pack_files_by_category "$git_root" '\.py$' "$python_output_file" "Python Files"
    pack_files_by_category "$git_root" '(\.yaml|\.yml|\.prompt)$' "$config_output_file" "YAML & Prompt Files"
    pack_files_by_category "$git_root" '\.sh$' "$shell_output_file" "Shell Scripts"

    echo "--------------------------------------------------"
    echo "🎉 Project packaging complete! The following files have been created in '$git_root':"
    [ -f "${git_root}/${structure_output_file}" ] && echo "   - ${structure_output_file}"
    [ -f "${git_root}/${python_output_file}" ] && echo "   - ${python_output_file}"
    [ -f "${git_root}/${config_output_file}" ] && echo "   - ${config_output_file}"
    [ -f "${git_root}/${shell_output_file}" ] && echo "   - ${shell_output_file}"
}

main