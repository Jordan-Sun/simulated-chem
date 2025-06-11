from workload import Workload
from assignment import Assignment

import numpy as np
import pandas as pd
from pyscipopt import Model

def workload_per_processor(workload: Workload, original_assignment: Assignment) -> pd.DataFrame:
    """
    Computes the workload on each processor based on the original assignment.

    Args:
        workload (Workload): The workload to be processed.
        original_assignment (Assignment): The original assignment of tasks.

    Returns:
        pd.DataFrame: A DataFrame containing the workload on each processor.
    """
    # Vectorized computation of workload per processor per interval
    assignments = original_assignment.assignment.iloc[:, 0].to_numpy(dtype=int)
    workload_matrix = workload.workload.to_numpy(dtype=float)
    procs = original_assignment.processors
    intervals = workload.intervals
    # Use bincount for each interval to sum workloads per processor
    result = np.vstack([
        np.bincount(assignments, weights=workload_matrix[:, interval], minlength=procs)
        for interval in range(intervals)
    ]).T
    return pd.DataFrame(result, index=range(procs), columns=workload.workload.columns)


def mip_group_workload(processor_workload: pd.DataFrame, num_groups: int) -> list[list[int]]:
    """
    Returns a list of processor groups based on the workload that can be set using set_processor_groups.
    Solves for the best grouping of processors into `num_groups` groups through solving a mixed-integer programming (MIP) problem for optimality.
    """
    procs = processor_workload.shape[0]
    proc_total = processor_workload.sum(axis=1).to_numpy(dtype=float)
    # Make sure it is fully divisible
    if procs % num_groups != 0:
        raise ValueError("Number of processors is not divisible by number of groups.")
    proc_per_group = np.ceil(procs / num_groups).astype(int)
    model = Model("ProcessorGroupingMIP")
    # Binary variables: x[i,g] = 1 if processor i in group g
    x = {}
    for i in range(procs):
        for g in range(num_groups):
            x[i, g] = model.addVar(vtype="B", name=f"x_{i}_{g}")
    # Continuous variable for the max group workload
    L = model.addVar(vtype="C", name="L")
    # Each processor assigned to exactly one group
    for i in range(procs):
        model.addCons(sum(x[i, g] for g in range(num_groups)) == 1)
    # Each group has exactly proc_per_group processors
    for g in range(num_groups):
        model.addCons(sum(x[i, g] for i in range(procs)) == proc_per_group)
    # Max group workload constraint
    for g in range(num_groups):
        model.addCons(sum(x[i, g] * proc_total[i] for i in range(procs)) <= L)
    model.setObjective(L, "minimize")
    model.optimize()
    groups = [[] for _ in range(num_groups)]
    if model.getStatus() != "optimal":
        print("Warning: MIP did not find an optimal solution.")
    for i in range(procs):
        for g in range(num_groups):
            if model.getVal(x[i, g]) > 0.5:
                groups[g].append(i)
    return groups

def greedy_group_workload(processor_workload: pd.DataFrame, num_groups: int) -> list[list[int]]:
    """
    Returns a list of processor groups based on the workload that can be set using set_processor_groups.
    Uses a greedy approach to group processors into `num_groups` groups based on their total workload.
    """
    procs = processor_workload.shape[0]
    proc_total = processor_workload.sum(axis=1).to_numpy(dtype=float)
    idx = np.argsort(-proc_total)
    groups = [[] for _ in range(num_groups)]
    group_loads = np.zeros(num_groups, dtype=float)
    for i in idx:
        g = np.argmin(group_loads)
        groups[g].append(int(i))
        group_loads[g] += float(proc_total[i])
    return groups

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python grouping.py <workload_file> <assignment_file> <num_groups>")
        sys.exit(1)

    workload_file = sys.argv[1]
    assignment_file = sys.argv[2]
    num_groups = int(sys.argv[3])

    workload = Workload.read_csv(workload_file)
    original_assignment = Assignment.read_csv(assignment_file)

    processor_workload = workload_per_processor(workload, original_assignment)
    print("Processor Workload:\n", processor_workload)

    greedy_groups = greedy_group_workload(processor_workload, num_groups)
    print("Greedy Groups:\n", greedy_groups)

    groups = mip_group_workload(processor_workload, num_groups)
    print("MIP Groups:\n", groups)
