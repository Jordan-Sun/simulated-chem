#!/bin/bash
if [[ -z $1 ]];
then 
    echo "usage: $0 <group>"
else
    for group in {0..11}
    do
        let index=$(($group*320 + $1))
        echo "Simulating $index."
        # python3 script.py simulate bound $index
        python3 script.py simulate proximity_y $index -1 2 3
        python3 script.py simulate proximity_x $index -1 2 3
        python3 script.py simulate proximity_matrix $index -1 2 3
        python3 script.py simulate simple_random $index 0 2 3
        python3 script.py simulate simple_random $index 1 2 3
        python3 script.py simulate simple_random $index 2 2 3
        python3 script.py simulate simple_random $index 3 2 3
        python3 script.py simulate simple_random $index 4 2 3
        python3 script.py simulate simple_random $index 5 2 3
        python3 script.py simulate simple_random $index 6 2 3
        python3 script.py simulate simple_random $index 7 2 3
        python3 script.py simulate simple_random $index 8 2 3
        python3 script.py simulate simple_random $index 9 2 3
        python3 script.py simulate balanced_random $index 0 2 3
        python3 script.py simulate balanced_random $index 1 2 3
        python3 script.py simulate balanced_random $index 2 2 3
        python3 script.py simulate balanced_random $index 3 2 3
        python3 script.py simulate balanced_random $index 4 2 3
        python3 script.py simulate balanced_random $index 5 2 3
        python3 script.py simulate balanced_random $index 6 2 3
        python3 script.py simulate balanced_random $index 7 2 3
        python3 script.py simulate balanced_random $index 8 2 3
        python3 script.py simulate balanced_random $index 9 2 3
        python3 script.py simulate greedy $index -1 2 3
    done
fi