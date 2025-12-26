#!/bin/bash

# Script to combine all Python files in the project into a single file
# Usage: ./combine_python_files.sh

# Determine the project root directory (where the script is located)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_FILE="$PROJECT_ROOT/sys/all_files_in_one.py"

# Create the sys directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Find all Python files in the project and append them to the output file
find "$PROJECT_ROOT" -name "*.py" -type f -not -path "$OUTPUT_FILE" | while read -r file; do
    echo "# File: $file" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"  # Add an empty line between files for readability
done

echo "All Python files have been combined into $OUTPUT_FILE"