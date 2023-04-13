#!/bin/bash

if [[ -z $1 ]];
then 
    echo "usage: $0 <index>"
else
    for offset in {1..20}
    do
        let index=$(($1*60+$offset))
        let extra=-1
        for send in 1 5
        do
            python3 script.py complicated greedy_weak_neighbor_dependent_unicast $index $extra $send $send
            python3 script.py simulate greedy_weak_neighbor_dependent_unicast $index $extra $send $send
            let recv=$send*2
            python3 script.py complicated greedy_weak_neighbor_dependent_unicast $index $extra $send $recv
            python3 script.py simulate greedy_weak_neighbor_dependent_unicast $index $extra $send $recv
        done
    done
fi