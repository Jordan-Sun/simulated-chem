#!/bin/bash
if [[ -z $1 ]];
then 
    echo "usage: $0 <group>"
else
    for group in {0..3}
    do
        let index=$(($group*960 + $1))
        echo "Running $index."
        # sendCost = 2 recvCost = 3
        # python3 script.py lp_compute_commute solve $index [extra](-1) [sendCost] [recvCost]
        python3 script.py lp_compute_commute solve $index 0 2 3
        python3 script.py lp_compute_commute solve $index 1 2 3
        python3 script.py lp_compute_commute solve $index 2 2 3
        python3 script.py lp_compute_commute solve $index 3 2 3
        python3 script.py lp_compute_commute solve $index 4 2 3
        python3 script.py lp_compute_commute solve $index 5 2 3
        python3 script.py lp_compute_commute solve $index 6 2 3
        python3 script.py lp_compute_commute solve $index 7 2 3
        python3 script.py lp_compute_commute solve $index 8 2 3
        python3 script.py lp_compute_commute solve $index 9 2 3
        
    done
fi