"""
greedy.py
Algorithms for the dynamic constrained reassignment problem:
    - Assignment can change from interval to interval, thus algorithms only consider the current interval
    - The number of processors that columns can be reassigned to is constrained to 1.
"""
from workload import Workload
from assignment import Assignment

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# import time

# Helper method to swap columns between two processors with greedy heuristic


def greedy_swap(setA: np.ndarray, setB: np.ndarray, threshold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort both sets by the workload. Then repeatedly swap the most costly column from the higher cost set with the least costly column from the lower cost set until the costs of the higher cost set would be less than the lower cost set. When that happens, swap with the column that would minimize the difference between the costs as the last swap.
    """
    # Convert inputs if necessary
    if not isinstance(setA, np.ndarray):
        setA = np.array(setA, dtype=int)
        print("Warning: setA was not a numpy array")
    if not isinstance(setB, np.ndarray):
        setB = np.array(setB, dtype=int)
        print("Warning: setB was not a numpy array")
    # Find the processor with the higher cost
    costA = np.sum(setA[:, 1])
    costB = np.sum(setB[:, 1])

    # Sort high cost set in descending order
    # Sort low cost set in ascending order
    # Keep track of the indices of the sorted set

    if costA > costB:
        set_high = setA[setA[:, 1].argsort()[::-1]]
        cost_high = costA
        set_low = setB[setB[:, 1].argsort()]
        cost_low = costB
    else:
        set_high = setB[setB[:, 1].argsort()[::-1]]
        cost_high = costB
        set_low = setA[setA[:, 1].argsort()]
        cost_low = costA

    # Greedy swap
    swap_index = 0
    ids_high = []
    ids_low = []
    # While the cost of the high cost set is greater than the low cost set or one of the sets is exhausted
    while cost_high > cost_low + threshold and swap_index < set_high.shape[0] and swap_index < set_low.shape[0]:
        # Swap the most costly column from the high cost set with the least costly column from the low cost set
        # Update the costs
        cost_high -= set_high[swap_index, 1]
        cost_low += set_high[swap_index, 1]
        # Update the indicies to be swapped
        ids_high.append(set_low[swap_index, 0])
        ids_low.append(set_high[swap_index, 0])
        swap_index += 1

    # Revert last swap
    if swap_index > 0:
        swap_index -= 1
        cost_high += set_high[swap_index, 1]
        cost_low -= set_high[swap_index, 1]
        ids_high.pop()
        ids_low.pop()

    # Compute the difference between each remaining pair of columns and make one last swap only if the threshold is 0.
    if threshold == 0:
        differences = np.zeros(
            (set_high.shape[0] - swap_index, set_low.shape[0] - swap_index))
        min_diff = np.inf
        min_pair = None
        for i in range(swap_index, set_high.shape[0]):
            for j in range(swap_index, set_low.shape[0]):
                differences[i - swap_index, j - swap_index] = abs(
                    (cost_high - cost_low) - (set_high[i, 1] - set_low[j, 1]))
                if differences[i - swap_index, j - swap_index] < min_diff:
                    min_diff = differences[i - swap_index, j - swap_index]
                    min_pair = (i, j)

        # Fill in the remaining indicies
        for i in range(swap_index, set_high.shape[0]):
            if i != min_pair[0]:
                ids_high.append(set_high[i, 0])
            else:
                ids_high.append(set_low[min_pair[1], 0])
        for j in range(swap_index, set_low.shape[0]):
            if j != min_pair[1]:
                ids_low.append(set_low[j, 0])
            else:
                ids_low.append(set_high[min_pair[0], 0])
    else:
        # Fill in the remaining indicies
        for i in range(swap_index, set_high.shape[0]):
            ids_high.append(set_high[i, 0])
        for j in range(swap_index, set_low.shape[0]):
            ids_low.append(set_low[j, 0])

    # Return the indicies of the columns to swap
    if costA > costB:
        return np.array(ids_high, dtype=int), np.array(ids_low, dtype=int)
    else:
        return np.array(ids_low, dtype=int), np.array(ids_high, dtype=int)


# Helper method to swap columns between two processors with dynamic programming
def dp_swap(setA: np.ndarray, setB: np.ndarray, threshold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs an DP table that minimizes the absolute difference between the sums of the columns in setA and setB after swapping.
    """
    # Convert inputs if necessary
    if not isinstance(setA, np.ndarray):
        setA = np.array(setA, dtype=int)
        print("Warning: setA was not a numpy array")
    if not isinstance(setB, np.ndarray):
        setB = np.array(setB, dtype=int)
        print("Warning: setB was not a numpy array")

    N = setA.shape[0]
    # Vectorized difference computation
    differences = setA[:, 1] - setB[:, 1]
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
        swap_alg=greedy_swap,
        result_path: Optional[str] = None,
        threshold_factor: float = 0
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
            print(
                f'Error {e} occurred while reading the assignment at {result_path} for interval {interval}')

    print(f'Starting interval {interval}')
    # Use numpy to avoid per-column loops
    assignment_ranks = original_assignment.assignment['KppRank'].to_numpy()
    workload_interval = workload.workload.iloc[:, interval].to_numpy()
    processor_costs = np.bincount(
        assignment_ranks, weights=workload_interval, minlength=original_assignment.processors)
    processor_columns = {proc: np.column_stack((np.nonzero(assignment_ranks == proc)[0],
                                                workload_interval[assignment_ranks == proc]))
                         for proc in range(original_assignment.processors)}

    sorted_processors = np.argsort(processor_costs)

    # multiply the maximum cost by the threshold percentage to get the threshold
    if threshold_factor > 0:
        max_cost = processor_costs.max()
        threshold = int(max_cost * threshold_factor / 100)
        print(f"Threshold for interval {interval} is {threshold} ({(threshold_factor * 100):.2f}% of max cost {max_cost})")
    else:
        threshold = 0

    # prepare processor pairs based on sorted costs
    half = len(sorted_processors) // 2
    tasks = np.array([(processor_columns[sorted_processors[i]], processor_columns[sorted_processors[-i - 1]])
                      for i in range(half)], dtype=object)
    pairs_index = np.array([(sorted_processors[i], sorted_processors[-i - 1])
                            for i in range(half)], dtype=int)

    assignments = np.full(workload.samples, -1, dtype=int)

    results = [swap_alg(pairs_A, pairs_B, threshold) for pairs_A, pairs_B in tasks]

    # update assignments based on results using indices
    for i in range(len(results)):
        proc_a, proc_b = pairs_index[i]
        set_A, set_B = results[i]
        for col in set_A:
            assignments[col] = proc_a
        for col in set_B:
            assignments[col] = proc_b

    # compute updated costs using numpy grouping
    updated_costs = np.bincount(
        assignments, weights=workload_interval, minlength=original_assignment.processors)
    peak = updated_costs.max()
    trough = updated_costs.min()
    print(f"Interval {interval} diff: {peak} - {trough} = {peak - trough}")

    if result_path is not None:
        Assignment(pd.DataFrame(assignments, columns=[
                   'KppRank'])).write_csv(result_path)
    return Assignment(pd.DataFrame(assignments, columns=['KppRank']))

# Greedy heuristic but limit the swap to between local processors


def greed_heuristic_local(
        workload: Workload,
        original_assignment: Assignment,
        interval: int = 0,
        swap_alg=dp_swap,
        result_path: Optional[str] = None,
        threshold_factor: float = 0
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
            print(
                f'Error {e} occurred while reading the assignment at {result_path} for interval {interval}')

    print(f'Starting interval {interval}')
    # Use numpy to avoid per-column loops
    assignment_ranks = original_assignment.assignment['KppRank'].to_numpy()
    workload_interval = workload.workload.iloc[:, interval].to_numpy()

    assignments = np.full(workload.samples, -1, dtype=int)

    group_id = 0
    for group in original_assignment.processor_groups:
        processor_costs = np.bincount(
            assignment_ranks, weights=workload_interval, minlength=original_assignment.processors)
        processor_columns = {proc: np.column_stack((np.nonzero(assignment_ranks == proc)[0],
                                                    workload_interval[assignment_ranks == proc]))
                             for proc in group}

        sorted_processors = sorted(
            group, key=lambda proc: processor_costs[proc])

        # multiply the maximum cost by the threshold percentage to get the threshold
        if threshold_factor > 0:
            max_cost = processor_costs.max()
            threshold = int(max_cost * threshold_factor)
            group_id += 1
            print(
                f"Threshold for interval {interval}, group {group_id} is {threshold} ({(threshold_factor * 100):.2f}% of max cost {max_cost})")
        else:
            threshold = 0

        # prepare processor pairs based on sorted costs
        half = len(sorted_processors) // 2
        tasks = np.array([(processor_columns[sorted_processors[i]], processor_columns[sorted_processors[-i - 1]])
                          for i in range(half)], dtype=object)
        pairs_index = np.array([(sorted_processors[i], sorted_processors[-i - 1])
                                for i in range(half)], dtype=int)

        results = [swap_alg(pairs_A, pairs_B, threshold) for pairs_A, pairs_B in tasks]

        # update assignments based on results using indices
        for i in range(len(results)):
            proc_a, proc_b = pairs_index[i]
            set_A, set_B = results[i]
            for col in set_A:
                assignments[col] = proc_a
            for col in set_B:
                assignments[col] = proc_b

    # compute updated costs using numpy grouping
    updated_costs = np.bincount(
        assignments, weights=workload_interval, minlength=original_assignment.processors)
    peak = updated_costs.max()
    trough = updated_costs.min()
    print(f"Interval {interval} diff: {peak} - {trough} = {peak - trough}")

    if result_path is not None:
        Assignment(pd.DataFrame(assignments, columns=[
                   'KppRank'])).write_csv(result_path)
    return Assignment(pd.DataFrame(assignments, columns=['KppRank']))


# If ran as main, test the heuristic
if __name__ == "__main__":
    print("This script is now a library. Use run_greedy_interval.py and combine_assignments.py for batch processing.")
