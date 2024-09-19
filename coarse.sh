#!/bin/bash

# Index is ignored
index=1

# Loop through the costs
# for iterations in 1 2
for cost in 1.005
do
  # Run the first python command
  python3 script.py dynamic dynamic "$index" 1000 1 "$cost"
  # python3 script.py dynamic limited_dynamic "$index" "$iterations" 1 1.001
  # Run the second python command
  python3 script.py simulate dynamic_reassignment "$index" "$cost"
done