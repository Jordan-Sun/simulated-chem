#!/bin/bash

if [[ -z $1 ]];
then 
    echo "usage: $0 <index>"
else
    for group in {1..12}
    do
        let index=$(($group*320+$1))
        let extra=-1
        for send in 1 5
        do
            python3 script.py complicated greedy_independent_broadcast $index $extra $send $send
            python3 script.py simulate greedy_independent_broadcast $index $extra $send $send
            python3 script.py complicated greedy_independent_unicast $index $extra $send $send
            python3 script.py simulate greedy_independent_unicast $index $extra $send $send
            python3 script.py complicated greedy_dependent_broadcast $index $extra $send $send
            python3 script.py simulate greedy_dependent_broadcast $index $extra $send $send
            python3 script.py complicated greedy_dependent_unicast $index $extra $send $send
            python3 script.py simulate greedy_dependent_unicast $index $extra $send $send
            let recv=$send*2
            python3 script.py complicated greedy_independent_broadcast $index $extra $send $recv
            python3 script.py simulate greedy_independent_broadcast $index $extra $send $recv
            python3 script.py complicated greedy_independent_unicast $index $extra $send $recv
            python3 script.py simulate greedy_independent_unicast $index $extra $send $recv
            python3 script.py complicated greedy_dependent_broadcast $index $extra $send $recv
            python3 script.py simulate greedy_dependent_broadcast $index $extra $send $recv
            python3 script.py complicated greedy_dependent_unicast $index $extra $send $recv
            python3 script.py simulate greedy_dependent_unicast $index $extra $send $recv
        done
    done
fi