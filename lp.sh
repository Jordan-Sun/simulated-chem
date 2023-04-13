#!/bin/bash
if [[ -z $1 ]];
then 
    echo "usage: $0 <group>"
else
    for group in {0..11}
    do
        let index=$(($group*320 + $1))
        # echo "Solving LP $index"
        # python3 script.py lp solve $index
        # echo "Post Processing LP $index"
        # python3 script.py lp max $index
        # python3 script.py lp random $index 0
        # python3 script.py lp random $index 1
        # python3 script.py lp random $index 2
        # python3 script.py lp random $index 3
        # python3 script.py lp random $index 4
        # python3 script.py lp random $index 5
        # python3 script.py lp random $index 6
        # python3 script.py lp random $index 7
        # python3 script.py lp random $index 8
        # python3 script.py lp random $index 9
        echo "Simulating LP $index"
        let send=1
        let recv=2
        python3 script.py simulate lp_max $index -1 $send $recv
        python3 script.py simulate lp_random $index 0 $send $recv
        python3 script.py simulate lp_random $index 1 $send $recv
        python3 script.py simulate lp_random $index 2 $send $recv
        python3 script.py simulate lp_random $index 3 $send $recv
        python3 script.py simulate lp_random $index 4 $send $recv
        python3 script.py simulate lp_random $index 5 $send $recv
        python3 script.py simulate lp_random $index 6 $send $recv
        python3 script.py simulate lp_random $index 7 $send $recv
        python3 script.py simulate lp_random $index 8 $send $recv
        python3 script.py simulate lp_random $index 9 $send $recv
        let send=5
        let recv=5
        python3 script.py simulate lp_max $index -1 $send $recv
        python3 script.py simulate lp_random $index 0 $send $recv
        python3 script.py simulate lp_random $index 1 $send $recv
        python3 script.py simulate lp_random $index 2 $send $recv
        python3 script.py simulate lp_random $index 3 $send $recv
        python3 script.py simulate lp_random $index 4 $send $recv
        python3 script.py simulate lp_random $index 5 $send $recv
        python3 script.py simulate lp_random $index 6 $send $recv
        python3 script.py simulate lp_random $index 7 $send $recv
        python3 script.py simulate lp_random $index 8 $send $recv
        python3 script.py simulate lp_random $index 9 $send $recv
        let send=5
        let recv=10
        python3 script.py simulate lp_max $index -1 $send $recv
        python3 script.py simulate lp_random $index 0 $send $recv
        python3 script.py simulate lp_random $index 1 $send $recv
        python3 script.py simulate lp_random $index 2 $send $recv
        python3 script.py simulate lp_random $index 3 $send $recv
        python3 script.py simulate lp_random $index 4 $send $recv
        python3 script.py simulate lp_random $index 5 $send $recv
        python3 script.py simulate lp_random $index 6 $send $recv
        python3 script.py simulate lp_random $index 7 $send $recv
        python3 script.py simulate lp_random $index 8 $send $recv
        python3 script.py simulate lp_random $index 9 $send $recv
    done
fi