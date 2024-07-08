#!/bin/bash

# Index is ignored
index=1

# Loop through the costs
for cost in 0
do
  # Run the first python command
  python3 script.py complicated balance_greedy "$index" "$cost"
  # Run the second python command
  python3 script.py simulate balance_greedy "$index" "$cost"
done