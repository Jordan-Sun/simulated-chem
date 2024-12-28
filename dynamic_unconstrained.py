"""
dynamic_unconstrained.py
Algorithms for the dynamic unconstrained reassignment problem:
    - Assignment can change from interval to interval, thus algorithms only consider the current interval
    - The number of processors that columns can be reassigned to is unconstrained.
"""
from workload import Workload
from assignment import Assignment

from pyscipopt import Model
import os
import pandas as pd

# Optimal one-to-many dynamic problem reassignment solution through solving MILP problem
def dynamic_MILP(
        workload: Workload,
        original_assignment: Assignment,
        interval: int = 0
) -> Assignment:
    """
    Goal: minimize L
    Constraints:
        L geq sum_{c in C, i in P} x_{c, i, k} w_{c, i} forall k in P  
        sum_{c in C, i in P} x_{c, i, k} = sum_{c in C, j in P} x_{c, k, j} forall k in P 
        sum_{j in P} x_{c, i, j} = 1 forall c in C, i in P
    Variables:
        L in R
        x_{c,i,j} in {0,1} forall c in C, (i, j) in P
    """