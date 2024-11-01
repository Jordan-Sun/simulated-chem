#!/bin/bash

if [[ -z $1 ]];
then 
    echo "usage: $0 <index>"
else
    index=$1
    function="dynamic"

    echo "Solving..."
    python3 script.py dynamic $function $index
    echo "Simulating..."
    python3 quick_sim.py nc4/p_24/$function
fi