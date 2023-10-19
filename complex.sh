#!/bin/bash

if [[ -z $1 ]];
then 
    echo "usage: $0 <index>"
else
    let index=1
    tunes=("0.0" "0.2" "0.4" "0.6" "0.8" "1.0")
    tune_index=$(($1 / 5))
    seeds=(-1 0 1 2 3)
    seed_index=$(($1 % 5))


    echo "Solving..."
    python3 script.py complicated mosaic_greedy $index ${seeds[$seed_index]} 1 1 ${tunes[$tune_index]}
    echo "Simulating..."
    for cost in 10 20
    do
        python3 script.py simulate mosaic_greedy $index ${seeds[$seed_index]} $cost $cost ${tunes[$tune_index]}
    done
fi