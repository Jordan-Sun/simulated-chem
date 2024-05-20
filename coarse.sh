#!/bin/bash

# Loop for index from 200 to 400, going up by 1
for index in $(seq 263 1 400); do
  # Loop for constant from 0 to 1, going up by 0.1
  for constant in $(seq 0 0.1 1); do
    # Loop for coarse from 2 to 5, going up by 1
    for coarse in $(seq 2 1 5); do
      # Run the first python command
      python3 script.py complicated greedy_prioritize_communication 50 "$index" 10 20 "$constant" "$coarse"
      # Run the second python command
      python3 script.py simulate greedy_prioritize_communication 50 "$index" 10 20 "$constant" "$coarse"
    done
  done
done