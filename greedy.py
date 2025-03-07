"""
greedy.py
Algorithms for the dynamic constrained reassignment problem:
    - Assignment can change from interval to interval, thus algorithms only consider the current interval
    - The number of processors that columns can be reassigned to is constrained to 1.
"""
from workload import Workload
from assignment import Assignment

import os
import multiprocessing
import numpy as np
import pandas as pd
from typing import List, Tuple
from functools import partial


# Helper method to swap columns between two processors
def swap_columns(setA: np.ndarray, setB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updated swap_columns that expects setA and setB as numpy arrays of shape (N,2)
    where first column is id and second column is integer value.
    Returns two numpy arrays containing column ids for setA and setB respectively.
    """
    # Convert inputs if necessary
    if not isinstance(setA, np.ndarray):
        setA = np.array(setA, dtype=int)
    if not isinstance(setB, np.ndarray):
        setB = np.array(setB, dtype=int)
    
    N = setA.shape[0]
    # Vectorized difference computation
    differences = setA[:,1] - setB[:,1]
    total_diff = int(np.sum(np.abs(differences)))
    offset = total_diff  # offset for DP array index

    # Initialize DP arrays
    # dp_prev[i, j] stores the previous DP index (j index before adding current difference)
    # dp_choice[i, j] stores +1 if plus was chosen or -1 for minus at step i.
    magic_number = -9999  # arbitrary number to represent invalid state
    dp_prev = np.full((N+1, 2*total_diff+1), magic_number, dtype=int)
    dp_choice = np.zeros((N+1, 2*total_diff+1), dtype=int)
    # Mark starting state (sum=0 mapped to index offset)
    dp_prev[0, offset] = offset  

    # DP loop: state index ranges 0 to 2*total_diff (with offset representing 0)
    for i in range(1, N+1):
        di = int(differences[i-1])
        for j in range(2*total_diff+1):
            if dp_prev[i-1, j] != magic_number:
                # Option 1: choose plus: new sum = (j - offset) + di
                new_idx = j + di
                if 0 <= new_idx < 2*total_diff+1 and dp_prev[i, new_idx] == magic_number:
                    dp_prev[i, new_idx] = j
                    dp_choice[i, new_idx] = 1  # plus chosen

                # Option 2: choose minus: new sum = (j - offset) - di
                new_idx = j - di
                if 0 <= new_idx < 2*total_diff+1 and dp_prev[i, new_idx] == magic_number:
                    dp_prev[i, new_idx] = j
                    dp_choice[i, new_idx] = -1  # minus chosen

    # Find final state with minimal absolute sum
    best_idx = None
    best_abs = None
    for j in range(2*total_diff+1):
        if dp_prev[N, j] != magic_number:
            current_abs = abs(j - offset)
            if best_abs is None or current_abs < best_abs:
                best_abs = current_abs
                best_idx = j

    # Backtrack decisions
    ids_setA = []
    ids_setB = []
    cur_idx = best_idx
    for i in range(N, 0, -1):
        choice = dp_choice[i, cur_idx]
        if choice == 1:
            # plus: take setA's id to setA and setB's id to setB
            ids_setA.append(setA[i-1, 0])
            ids_setB.append(setB[i-1, 0])
        else:
            # minus: swap choices
            ids_setA.append(setB[i-1, 0])
            ids_setB.append(setA[i-1, 0])
        cur_idx = dp_prev[i, cur_idx]  # back-pointer

    # Reverse to restore order
    ids_setA = np.array(ids_setA[::-1], dtype=int)
    ids_setB = np.array(ids_setB[::-1], dtype=int)
    return ids_setA, ids_setB

# Greedy one-to-one dynamic reassignment solution through greedy heuristic
def greed_heuristic(
        workload: Workload,
        original_assignment: Assignment,
        interval: int = 0,
        pool_size: int = 1,
        result_path: str = None
) -> Assignment:
    # Check if an assignment is already at the result path
    if result_path is not None:
        try:
            assignment = Assignment.read_csv(result_path)
            print(f'Skipping interval {interval}')
            return assignment
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f'Error {e} occurred while reading the assignment at {result_path} for interval {interval}')
    
    print(f'Starting interval {interval}')
    # Use numpy to avoid per-column loops
    assignment_ranks = original_assignment.assignment['KppRank'].to_numpy()
    workload_interval = workload.workload.iloc[:, interval].to_numpy()
    processor_costs = np.bincount(assignment_ranks, weights=workload_interval, minlength=original_assignment.processors)
    processor_columns = {proc: list(zip(np.nonzero(assignment_ranks == proc)[0],
                                        workload_interval[assignment_ranks == proc]))
                         for proc in range(original_assignment.processors)}
    
    sorted_processors = np.argsort(processor_costs)
    
    # prepare processor pairs based on sorted costs
    tasks = []
    pairs_index = []
    half = len(sorted_processors) // 2
    for i in range(half):
        proc_a = sorted_processors[i]
        proc_b = sorted_processors[-i - 1]
        tasks.append((processor_columns[proc_a], processor_columns[proc_b]))
        pairs_index.append((proc_a, proc_b))
    
    assignments = np.full(workload.samples, -1, dtype=int)
    if pool_size == 1:
        results = [swap_columns(pairs_A, pairs_B) for pairs_A, pairs_B in tasks]
    else:
        with multiprocessing.Pool(processes=pool_size) as pool:
            results = pool.starmap(swap_columns, tasks)
    
    # update assignments based on results using indices
    for i in range(len(results)):
        proc_a, proc_b = pairs_index[i]
        set_A, set_B = results[i]
        for col in set_A:
            assignments[col] = proc_a
        for col in set_B:
            assignments[col] = proc_b
    
    # compute updated costs using numpy grouping
    updated_costs = np.bincount(assignments, weights=workload_interval, minlength=original_assignment.processors)
    peak = updated_costs.max()
    trough = updated_costs.min()
    print(f"Interval {interval} diff: {peak} - {trough} = {peak - trough}")
    
    if result_path is not None:
        Assignment(pd.DataFrame(assignments, columns=['KppRank'])).write_csv(result_path)
    return Assignment(pd.DataFrame(assignments, columns=['KppRank']))


# If ran as main, test the heuristic
if __name__ == "__main__":

    # Test the heuristic at c24 resolution at 6 processors
    res = 48
    workload = Workload.read_csv(f"test/workloads/c{res}.csv")
    procs = 36
    original_assignment = Assignment.read_csv(
        f"test/og_assignments/c{res}_p{procs}.csv")
    base = f"test/greedy/c{res}_p{procs}"
    os.makedirs(base, exist_ok=True)
    assignments = []

    # Multiprocessing doesn't work quite well, use batching instead
    total_batches = 8
    pool_size = 8
    if len(os.sys.argv) > 1:
        batch = int(os.sys.argv[1])
    else:
        batch = None

    # Run the heuristic for each interval
    os.makedirs(f'{base}/intervals', exist_ok=True)
    for interval in range(workload.intervals):
        if batch:
            # Skip if not in the batch
            if interval % total_batches != batch:
                continue
        assignments.append(greed_heuristic(
            workload, original_assignment, interval, pool_size, f'{base}/intervals/{interval}.csv'))
    
    # Concatenate the assignments
    print("Concatenating assignments")
    assignment = Assignment.concatenate(assignments)
    assignment.write_csv(f'{base}/assignment.csv')
