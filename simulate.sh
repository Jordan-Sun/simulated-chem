#!/bin/bash
if [[ -z $1 ]];
then 
    echo "usage: $0 <index>"
else
    let index=$1
    # Simulation that does not require assignments
    echo "Doing basic simulations"
    python3 script.py simulate bound $index &
    python3 script.py simulate original $index &

    # Wait for all assignments to finish
    wait

    # Assignment loop
    echo "Doing greedy assignments"
    python3 script.py complicated greedy $index &
    python3 script.py complicated greedy $index -1 1 1 1 8 &
    python3 script.py complicated mosaic_greedy $index &

    # Wait for all assignments to finish
    wait

    # Simulation loop
    echo "Doing greedy simulations"
    python3 script.py simulate greedy $index &
    python3 script.py simulate greedy $index -1 1 1 1 8 &
    python3 script.py simulate mosaic_greedy $index &

    # Wait for all simulations to finish
    wait

    # Assignment loop
    echo "Doing simple random assignments"
    for offset in {0..9}
    do
        python3 script.py simple simple_random $index $offset 1 1 &
    done

    # Wait for all assignments to finish
    wait

    # Simulation loop
    echo "Doing random assignment simulations"

    for offset in {0..9}
    do
        python3 script.py simulate simple_random $index $offset 1 1 &
    done

    # Wait for all simulations to finish
    wait
fi