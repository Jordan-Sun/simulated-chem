#!/bin/bash

for i in {0..10}; do
    c=$(echo "scale=1; $i / 10" | bc -l)
    python3 script.py complicated greedy_prioritize_communication 50 3 10 10 $c
    python3 script.py simulate greedy_prioritize_communication 50 3 10 10 $c
done
