#!/bin/bash

num_batches=8

for ((i=0; i<=num_batches-1; i++))
do
    python greedy.py $i &
done

wait