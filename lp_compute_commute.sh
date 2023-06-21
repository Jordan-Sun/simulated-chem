#!/bin/bash

sendCost=2
recvCost=3
if [[ -z $1 ]];
then 
    echo "usage: $0 <group>"
else
    for group in {0..3}
    do
        let index=$(($group*100 + $1))
        echo "Running $index."
        # sendCost = 2 recvCost = 3
        # python3 script.py lp_compute_commute solve $index [extra](-1) [sendCost] [recvCost]
        for extra in {0..9}; do
            # Execute the command with the fixed index value
            python3 script.py lp_compute_commute solve $index $extra $sendCost $recvCost
            python3 script.py lp_compute_commute max $index $extra $sendCost $recvCost
            python3 script.py lp_compute_commute random $index $extra $sendCost $recvCost
            python3 script.py simulate lp_compute_commute_random $index $extra $sendCost $recvCost
            python3 script.py simulate lp_compute_commute_max $index $extra $sendCost $recvCost
        done
    done
fi