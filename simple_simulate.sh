#!/bin/bash
if [[ -z $1 ]];
then 
    echo "usage: $0 <index>"
else
    echo "Index $1"
    python3 script.py simple proximity_y $1
    python3 script.py simple proximity_x $1
    python3 script.py simple proximity_matrix $1
    python3 script.py simple simple_random $1 0
    python3 script.py simple simple_random $1 1
    python3 script.py simple simple_random $1 2
    python3 script.py simple simple_random $1 3
    python3 script.py simple simple_random $1 4
    python3 script.py simple simple_random $1 5
    python3 script.py simple simple_random $1 6
    python3 script.py simple simple_random $1 7
    python3 script.py simple simple_random $1 8
    python3 script.py simple simple_random $1 9
    python3 script.py simple balanced_random $1 0
    python3 script.py simple balanced_random $1 1
    python3 script.py simple balanced_random $1 2
    python3 script.py simple balanced_random $1 3
    python3 script.py simple balanced_random $1 4
    python3 script.py simple balanced_random $1 5
    python3 script.py simple balanced_random $1 6
    python3 script.py simple balanced_random $1 7
    python3 script.py simple balanced_random $1 8
    python3 script.py simple balanced_random $1 9
    python3 script.py simulate bound $1
    python3 script.py simulate proximity_y $1
    python3 script.py simulate proximity_x $1
    python3 script.py simulate proximity_matrix $1
    python3 script.py simulate simple_random $1 0
    python3 script.py simulate simple_random $1 1
    python3 script.py simulate simple_random $1 2
    python3 script.py simulate simple_random $1 3
    python3 script.py simulate simple_random $1 4
    python3 script.py simulate simple_random $1 5
    python3 script.py simulate simple_random $1 6
    python3 script.py simulate simple_random $1 7
    python3 script.py simulate simple_random $1 8
    python3 script.py simulate simple_random $1 9
    python3 script.py simulate balanced_random $1 0
    python3 script.py simulate balanced_random $1 1
    python3 script.py simulate balanced_random $1 2
    python3 script.py simulate balanced_random $1 3
    python3 script.py simulate balanced_random $1 4
    python3 script.py simulate balanced_random $1 5
    python3 script.py simulate balanced_random $1 6
    python3 script.py simulate balanced_random $1 7
    python3 script.py simulate balanced_random $1 8
    python3 script.py simulate balanced_random $1 9
    python3 script.py simulate lp_max $index
    python3 script.py simulate lp_random $index 0
    python3 script.py simulate lp_random $index 1
    python3 script.py simulate lp_random $index 2
    python3 script.py simulate lp_random $index 3
    python3 script.py simulate lp_random $index 4
    python3 script.py simulate lp_random $index 5
    python3 script.py simulate lp_random $index 6
    python3 script.py simulate lp_random $index 7
    python3 script.py simulate lp_random $index 8
    python3 script.py simulate lp_random $index 9
fi
