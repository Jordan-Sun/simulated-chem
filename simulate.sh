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
        python3 script.py simulate proximity_y $index -1 1 2
        python3 script.py simulate proximity_x $index -1 1 2
        python3 script.py simulate proximity_matrix $index -1 1 2
        python3 script.py simulate simple_random $index 0 1 2
        python3 script.py simulate simple_random $index 1 1 2
        python3 script.py simulate simple_random $index 2 1 2
        python3 script.py simulate simple_random $index 3 1 2
        python3 script.py simulate simple_random $index 4 1 2
        python3 script.py simulate simple_random $index 5 1 2
        python3 script.py simulate simple_random $index 6 1 2
        python3 script.py simulate simple_random $index 7 1 2
        python3 script.py simulate simple_random $index 8 1 2
        python3 script.py simulate simple_random $index 9 1 2
        python3 script.py simulate balanced_random $index 0 1 2
        python3 script.py simulate balanced_random $index 1 1 2
        python3 script.py simulate balanced_random $index 2 1 2
        python3 script.py simulate balanced_random $index 3 1 2
        python3 script.py simulate balanced_random $index 4 1 2
        python3 script.py simulate balanced_random $index 5 1 2
        python3 script.py simulate balanced_random $index 6 1 2
        python3 script.py simulate balanced_random $index 7 1 2
        python3 script.py simulate balanced_random $index 8 1 2
        python3 script.py simulate balanced_random $index 9 1 2
        python3 script.py simulate greedy $index -1 1 2

        python3 script.py simulate proximity_y $index -1 5 5
        python3 script.py simulate proximity_x $index -1 5 5
        python3 script.py simulate proximity_matrix $index -1 5 5
        python3 script.py simulate simple_random $index 0 5 5
        python3 script.py simulate simple_random $index 1 5 5
        python3 script.py simulate simple_random $index 2 5 5
        python3 script.py simulate simple_random $index 3 5 5
        python3 script.py simulate simple_random $index 4 5 5
        python3 script.py simulate simple_random $index 5 5 5
        python3 script.py simulate simple_random $index 6 5 5
        python3 script.py simulate simple_random $index 7 5 5
        python3 script.py simulate simple_random $index 8 5 5
        python3 script.py simulate simple_random $index 9 5 5
        python3 script.py simulate balanced_random $index 0 5 5
        python3 script.py simulate balanced_random $index 1 5 5
        python3 script.py simulate balanced_random $index 2 5 5
        python3 script.py simulate balanced_random $index 3 5 5
        python3 script.py simulate balanced_random $index 4 5 5
        python3 script.py simulate balanced_random $index 5 5 5
        python3 script.py simulate balanced_random $index 6 5 5
        python3 script.py simulate balanced_random $index 7 5 5
        python3 script.py simulate balanced_random $index 8 5 5
        python3 script.py simulate balanced_random $index 9 5 5
        python3 script.py simulate greedy $index -1 5 5

        python3 script.py simulate proximity_y $index -1 5 10
        python3 script.py simulate proximity_x $index -1 5 10
        python3 script.py simulate proximity_matrix $index -1 5 10
        python3 script.py simulate simple_random $index 0 5 10
        python3 script.py simulate simple_random $index 1 5 10
        python3 script.py simulate simple_random $index 2 5 10
        python3 script.py simulate simple_random $index 3 5 10
        python3 script.py simulate simple_random $index 4 5 10
        python3 script.py simulate simple_random $index 5 5 10
        python3 script.py simulate simple_random $index 6 5 10
        python3 script.py simulate simple_random $index 7 5 10
        python3 script.py simulate simple_random $index 8 5 10
        python3 script.py simulate simple_random $index 9 5 10
        python3 script.py simulate balanced_random $index 0 5 10
        python3 script.py simulate balanced_random $index 1 5 10
        python3 script.py simulate balanced_random $index 2 5 10
        python3 script.py simulate balanced_random $index 3 5 10
        python3 script.py simulate balanced_random $index 4 5 10
        python3 script.py simulate balanced_random $index 5 5 10
        python3 script.py simulate balanced_random $index 6 5 10
        python3 script.py simulate balanced_random $index 7 5 10
        python3 script.py simulate balanced_random $index 8 5 10
        python3 script.py simulate balanced_random $index 9 5 10
        python3 script.py simulate greedy $index -1 5 10
    done
fi