#!/bin/bash
if [[ -z $1 ]];
then 
    echo "usage: $0 <index>"
else
    let index=$1

    echo "Doing animations"
    # python3 script.py animate original $index &
    python3 script.py animate greedy $index &
    python3 script.py animate mosaic_greedy $index &

    # Wait for all assignments to finish
    wait
fi