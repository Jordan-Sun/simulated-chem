#!/bin/bash

if [[ -z $1 ]];
then 
    echo "usage: $0 <index>"
else
    let index=$1

    echo "Solving..."
    python3 script.py dynamic limited_dynamic $index
    echo "Simulating..."
    python3 script.py simulate limited_dynamic $index
fi