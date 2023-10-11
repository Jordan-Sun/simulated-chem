#!/bin/bash

let index=1
echo "Solving..."
# python3 script.py simple proximity_matrix $index -1 0 0
python3 script.py complicated greedy $index -1 0 0
echo "Simulating..."
for cost in 20 50
do
    python3 script.py simulate proximity_matrix $index -1 $cost $cost
    python3 script.py simulate proximity_matrix $index -1 $cost $((2*$cost))
    python3 script.py simulate greedy $index -1 $cost $cost
    python3 script.py simulate greedy $index -1 $cost $((2*$cost))
done