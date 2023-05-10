#!/bin/bash
if [[ -z $1 ]];
then 
    echo "usage: $0 <index>"
else
    for offset in {1..20}
    do
        let index=$(($offset + $1))
        echo "Simulating $index."
        python3 script.py alt_simulate greedy $index -1 2 3
    done
fi
