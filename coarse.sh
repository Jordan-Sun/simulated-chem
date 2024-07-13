#!/bin/bash

# Index is ignored
index=1

# Loop through the costs
for cost in 100
do
  # Run the first python command
  python3 script.py dynamic dynamic_reassignment "$index" "$cost"
  # Run the second python command
  # python3 script.py simulate dynamic_reassignment "$index" "$cost"
done