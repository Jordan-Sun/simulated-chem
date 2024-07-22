#!/bin/bash

# Index is ignored
index=1

# Loop through the costs
for ratio in 1 1.001 1.0005 1.0001
do
  # Run the first python command
  python3 script.py dynamic limited_dynamic "$index" 0 1 "$ratio"
  # Run the second python command
  # python3 script.py simulate dynamic_reassignment "$index" "$cost"
done