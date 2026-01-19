#!/bin/bash

# Script to combine all Python files in the project into a single file with improved readability
# Usage: ./combine_python_files.sh

# Determine the project root directory (where the script is located)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Calculate minutes from start of day for postfix
CURRENT_HOUR=$(date +"%H")
CURRENT_MINUTE=$(date +"%M")
MINUTES_FROM_START_OF_DAY=$(echo "${CURRENT_HOUR} * 60 + ${CURRENT_MINUTE}" | bc)
DATE_POSTFIX=$(date +"%Y_%m_%d")_${MINUTES_FROM_START_OF_DAY}

OUTPUT_FILE="$PROJECT_ROOT/sys/all_files_in_one_$DATE_POSTFIX.py"

# Create the sys directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Find all Python files in the project and append them to the output file
find "$PROJECT_ROOT" -name "*.py" -type f -not -path "$OUTPUT_FILE" | sort | while read -r file; do
    # Check if the file is in the sys directory - if so, skip it
    if [ "$file" != "$OUTPUT_FILE" ] && [ "$(dirname "$file")" = "$SCRIPT_DIR" ]; then
        continue
    fi
    
    echo "" >> "$OUTPUT_FILE"
    echo "############################################################" >> "$OUTPUT_FILE"
    echo "# FILE: ${file#$PROJECT_ROOT/}" >> "$OUTPUT_FILE"
    echo "############################################################" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
done