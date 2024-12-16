#!/usr/bin/env bash

# This script processes all log files in the logs/ directory and 
# writes the extracted "solution" portion (from 'objective value:' 
# through the end of the file) into corresponding files in the solutions/ directory.

if [ $# -lt 1 ]; then
  echo "Usage: $0 <root_directory>"
  exit 1
fi

ROOTDIR="$1"

# Create the solutions folder if it doesn't exist
mkdir -p "$ROOTDIR/solutions"

# Iterate over each file in the logs folder
for logfile in "$ROOTDIR/logs/"*; do
  # Make sure it's a regular file
  if [ -f "$logfile" ]; then
    filename=$(basename "$logfile")
    echo "Extracting solution from $filename..."
    # Extract from 'objective value:' to the end of file
    sed -n '/objective value:/,$p' "$logfile" > "$ROOTDIR/solutions/$filename"
  fi
done

echo "Extraction done."
