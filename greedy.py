"""
greedy.py
Algorithms for the dynamic constrained reassignment problem:
    - Assignment can change from interval to interval, thus algorithms only consider the current interval
    - The number of processors that columns can be reassigned to is constrained to 1.
"""
from workload import Workload
from assignment import Assignment

from typing import List, Tuple
import pandas as pd

# Helper method to swap columns between two processors
def swap_columns(
        setA: List[Tuple[int, int]], setB: List[Tuple[int, int]]
) -> Tuple[List[int], List[int]]:
    # Convert to pairs
    N = len(setA)
    pairs = list(zip(setA, setB))  # Each pair is ((idA, valA), (idB, valB))

    # Convert the float values to integers
    differences = []
    total_diff = 0
    for (idA, valA), (idB, valB) in pairs:
        valA_int = int(valA)
        valB_int = int(valB)
        diff = valA_int - valB_int
        differences.append(diff)
        total_diff += abs(diff)

    # Initialize the DP table
    dp = [{} for _ in range(N + 1)]  # dp[i][s] = (prev_s, choice)
    dp[0][0] = None  # Starting point

    # Build the DP table
    for i in range(1, N + 1):
        di = differences[i - 1]
        dp_i = dp[i]
        dp_prev = dp[i - 1]
        for s in dp_prev:
            # Option 1: Assign di with +1 (valA to set A, valB to set B)
            s_new = s + di
            if s_new not in dp_i:
                dp_i[s_new] = (s, "+")
            # Option 2: Assign di with -1 (valA to set B, valB to set A)
            s_new_neg = s - di
            if s_new_neg not in dp_i:
                dp_i[s_new_neg] = (s, "-")

    # Find the minimal absolute sum
    min_abs_sum = None
    target_s = None
    for s in dp[N]:
        abs_s = abs(s)
        if min_abs_sum is None or abs_s < min_abs_sum:
            min_abs_sum = abs_s
            target_s = s

    # Reconstruct the solution
    ids_setA = []
    ids_setB = []
    s = target_s
    for i in range(N, 0, -1):
        prev_s, sign = dp[i][s]
        ((idA, valA), (idB, valB)) = pairs[i - 1]
        if sign == "+":
            # valA to set A, valB to set B
            ids_setA.append(idA)
            ids_setB.append(idB)
        else:
            # valA to set B, valB to set A
            ids_setA.append(idB)
            ids_setB.append(idA)
        s = prev_s  # Move to the previous state

    # Reverse the IDs to correct the order
    ids_setA.reverse()
    ids_setB.reverse()

    return ids_setA, ids_setB

# Greedy one-to-one dynamic reassignment solution through greedy heuristic
def greed_heuristic(
        workload: Workload,
        original_assignment: Assignment,
        interval: int = 0,
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
    # Pair the most costly and least costly processors# the cells on each processor
    processor_columns = []
    for _ in range(original_assignment.processors):
        processor_columns.append({})
    # the costs of each processor
    processor_costs = [0] * original_assignment.processors
    # the list of assignments
    assignments = [-1] * workload.samples
    # compute the costs of each processor given the original assignment
    for column in range(workload.samples):
        processor_columns[original_assignment.assignment['KppRank'][column]
                          ][column] = workload.workload.iloc[column, interval]
        processor_costs[original_assignment.assignment['KppRank'][column]] += workload.workload.iloc[column, interval]
    # sorted costs of processors
    sorted_processors = sorted(
        range(original_assignment.processors), key=lambda x: processor_costs[x])

    # pair the most costly and least costly processors and swap the columns between them
    for i in range(len(sorted_processors) // 2):
        print('Pairing processor ', sorted_processors[i], 'and processor ', sorted_processors[-i - 1], 'for interval', interval)
        # run through minimization algorithm
        set_A, set_B = swap_columns(processor_columns[sorted_processors[i]].items(), processor_columns[sorted_processors[-i - 1]].items())
        # assign the columns to the processors
        for column in set_A:
            assignments[column] = sorted_processors[i]
        for column in set_B:
            assignments[column] = sorted_processors[-i - 1]
        # update the processor costs
        processor_costs[sorted_processors[i]] = sum(workload.workload.iloc[column, interval] for column in set_A)
        processor_costs[sorted_processors[-i - 1]] = sum(workload.workload.iloc[column, interval] for column in set_B)
    # print the achieved result
    peak = max(processor_costs)
    trough = min(processor_costs)
    print(f"Interval {interval} diff: {peak} - {trough} = {peak - trough}")

    # write the assignment to the result path
    if result_path is not None:
        Assignment(pd.DataFrame(assignments, columns=['KppRank'])).write_csv(result_path)
    # return the assignments
    return Assignment(pd.DataFrame(assignments, columns=[interval]))


# If ran as main, test the heuristic
if __name__ == "__main__":
    import os  
    from functools import partial

    # Test the heuristic at c24 resolution at 6 processors
    res = 24
    workload = Workload.read_csv(f"test/workloads/c{res}.csv")
    procs = 6
    original_assignment = Assignment.read_csv(
        f"test/og_assignments/c{res}_p{procs}.csv")
    base = f"test/greedy/c{res}_p{procs}"
    os.makedirs(base, exist_ok=True)
    assignments = []

    # Multiprocessing doesn't work quite well, use batching instead
    total_batches = 8
    if len(os.sys.argv) > 1:
        batch = int(os.sys.argv[1])

    for interval in range(workload.intervals):
        if batch:
            # Skip if not in the batch
            if interval % total_batches != batch:
                continue
        assignments.append(greed_heuristic(
            workload, original_assignment, interval, f'{base}/assignment_{interval}.csv'))
    
    # Concatenate the assignments
    print("Concatenating assignments")
    assignment = Assignment.concatenate(assignments)
    assignment.write_csv(f'{base}/assignment.csv')
