#!/usr/bin/env bash

# This script processes all log files in the logs/ directory and 
# writes the extracted "solution" portion (from 'objective value:' 
# through the end of the file) into corresponding files in the solutions/ directory.

if [ $# -lt 1 ]; then
  echo "Usage: $0 <root_directory>"
  exit 1
fi

ROOTDIR="$1"
mkdir -p "$ROOTDIR/solutions"

for logfile in "$ROOTDIR/logs/"*; do
  if [ -f "$logfile" ]; then
    filename=$(basename "$logfile")
    echo "Extracting solution from $filename..."

    # Extract lines from 'objective value:' to the end
    extracted="$(sed -n '/objective value:/,$p' "$logfile")"

    # Skip if no lines were extracted
    if [ -z "$extracted" ]; then
      # Extract the last complete line of the file with numbers but not legends
      extracted="$(grep -E '^[[:space:]]*[0-9]+.*[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+\.[0-9]+%[[:space:]]*\|[[:space:]]*[0-9]+\.[0-9]+%[[:space:]]*$' "$logfile" | tail -n 1)"
      
      # Print status message
      echo "$filename status: $extracted"
      continue
    else
      # Print confirmation message
      echo "Extracted solution from $filename"
    fi

    # Write extracted lines to the corresponding file in solutions/
    echo "$extracted" > "$ROOTDIR/solutions/$filename"
  fi
done

echo "Extraction done."
