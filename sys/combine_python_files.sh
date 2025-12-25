#!/bin/bash

# Script to combine all Python files in the project into a single file
# Usage: ./combine_python_files.sh

# Set the output file path
OUTPUT_FILE="sys/all_files_in_one.py"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Find all Python files in the project and append them to the output file
find . -name "*.py" -type f -not -path "./sys/all_files_in_one.py" | while read -r file; do
    echo "# File: $file" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"  # Add an empty line between files for readability
done

echo "All Python files have been combined into $OUTPUT_FILE"