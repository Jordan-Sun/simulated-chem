#!/bin/bash

if [[ -z $1 ]];
then 
    echo "usage: $0 <index>"
else
    let index=1
    let cost=10
    seeds=(-1 0)
    seed_index=$(($1 % 2))
    coarses=(0 0 0)
    temp=$(($1 / 2))
    coarse_index=$(($temp % 3))
    tunes=("0.019" "0.021")
    tune_index=$(($temp / 3))


    echo "Solving..."
    python3 script.py complicated blocked_greedy $index ${seeds[$seed_index]} 1 1 ${tunes[$tune_index]} ${coarses[$coarse_index]}
    echo "Simulating..."
    python3 script.py simulate blocked_greedy $index ${seeds[$seed_index]} $cost $cost ${tunes[$tune_index]} ${coarses[$coarse_index]}
fi