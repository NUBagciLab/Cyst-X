#!/usr/bin/env bash

# Set default values if arguments are omitted
DEFAULT_SRC="./Internal 2 Classes"
DEFAULT_OUT="./Internal 2 Classes Calibrated"

SRC_DIR="${1:-$DEFAULT_SRC}"
OUT_DIR="${2:-$DEFAULT_OUT}"

# Strip any trailing slashes
SRC_DIR="${SRC_DIR%/}"
OUT_DIR="${OUT_DIR%/}"

echo "Source Directory: $SRC_DIR"
echo "Output Directory: $OUT_DIR"

# Find and iterate over all .xlsx files recursively
find "$SRC_DIR" -type f -name "*.xlsx" | while IFS= read -r input_file; do
    # Extract relative path from source directory
    rel_path="${input_file#"$SRC_DIR"/}"
    
    # Construct corresponding destination path
    output_file="$OUT_DIR/$rel_path"
    
    # Ensure destination subfolder exists
    mkdir -p "$(dirname "$output_file")"
    
    echo "Processing: $input_file -> $output_file"
    python analysis_internal.py -i "$input_file" -o "$output_file"
done