"""
dynamic_unconstrained.py
Algorithms for the dynamic unconstrained reassignment problem:
    - Assignment can change from interval to interval, thus algorithms only consider the current interval
    - The number of processors that columns can be reassigned to is constrained to 1.
"""
from workload import Workload
from assignment import Assignment

from pyscipopt import Model
import os
import pandas as pd

# Optimal one-to-one dynamic reassignment solution through solving MIQCP problem
def MIQCP(
        workload: Workload,
        original_assignment: Assignment,
        interval: int = 0,
        max_threads: int = 1,
        solution_file: str = None,
        redirect_output: bool = False,
        bias: float = 0.0
) -> Assignment:
    """
    Goal: minimize L + bias * sum_{c in C} sum_{k in P} x_{c, k}
    Constraints: where i != j != k
        L geq sum_{c in C} (1 - x_{c, k}) w_{c, k} + sum_{i in P} s_{i,k} sum_{c in C} x_{c, i} w_{c, i} forall k in P
        sum_{i in P} s_{i, k} sum_{c in C} x_{c, i} = sum_{j in P} s_{k, j} sum_{c in C} x_{c, k} forall k in P
        sum_{i in P} s_{i, k} = 1 forall k in P
        sum_{j in P} s_{k, j} = 1 forall k in P
    Variables: where i != j
        L in R
        x_{c,k} in {0,1} forall c in C, k in P
        s_{i,j} in {0,1} forall (i, j) in P
    """
    # Extract number of processors and cells
    P = original_assignment.processors
    C = workload.samples // P

    # Combine the workload matrix and the original assignment matrix to obtain the w_{c,k} values
    w = [[] for _ in range(P)]
    for sample in range(workload.samples):
        rank = original_assignment.assignment['KppRank'][sample]
        w[rank].append(workload.workload.iloc[sample, interval])

    # Create the solver model
    solver = Model("MIQCP")

    # Create the variables
    L = solver.addVar(vtype="C", name="L")
    x = {}
    s = {}
    for c in range(C):
        for k in range(P):
            x[c, k] = solver.addVar(vtype="B", name=f"x_{c}_{k}")
    for i in range(P):
        for j in range(P):
            if i != j:
                s[i, j] = solver.addVar(vtype="B", name=f"s_{i}_{j}")

    # Create the constraints forall k in P
    for k in range(P):
        # sum_{i in P} s_{i, k} = 1, i != k
        solver.addCons(sum(s[i, k] for i in range(P) if i != k) == 1)
        # sum_{j in P} s_{k, j} = 1, j != k
        solver.addCons(sum(s[k, j] for j in range(P) if j != k) == 1)
        # sum_{i in P} sum_{c in C} s_{i,k} x_{c, i} = sum_{j in P} sum_{c in C} s_{k, j} x_{c, k}, i != j != k
        solver.addCons(sum(s[i, k] * x[c, i] for i in range(P) if i != k for c in range(C))
                    == sum(s[k, j] * x[c, k] for j in range(P) if j != k for c in range(C)))
        # L geq sum_{c in C} (1 - x_{c, k}) w_{c, k} + sum_{i in P} s_{i,k} sum_{c in C} x_{c, i} w_{c, i}, i != k
        solver.addCons(L >= sum((1 - x[c, k]) * w[k][c] for c in range(C)) + sum(s[i, k] * sum(x[c, i] * w[i][c] for c in range(C)) for i in range(P) if i != k))

    # Add the objective function
    if bias == 0:
        solver.setObjective(L, "minimize")
    else:
        solver.setObjective(L + bias * sum(x[c, k] for c in range(C) for k in range(P)), "minimize")

    # If redirect_output is True, redirect the output to python, and let the user handle it
    if redirect_output:
        solver.redirectOutput()

    # If a solution file is provided, check if the solution is feasible
    feasible = False
    if solution_file is not None and os.path.exists(solution_file):
        # Read the solution from the file
        print(f"Reading solution from {solution_file}")
        sol = solver.readSolFile(solution_file)
        # Check if the solution is feasible
        feasible = solver.checkSol(sol)
        # Print an warning if the solution is not feasible
        if feasible:
            print(f"Solution for interval {interval} is feasible. Skipping the model.")
        else:
            print(f"Warning: Provided solution for interval {interval} is not feasible. Solving the model instead.")
            # Add the solution to the solver
            solver.addSol(sol)

    # Otherwise, solve the model
    if not feasible:
        # Cap memory usage to 80 GB
        solver.setRealParam("limits/memory", 80000.0)

        # If max_threads is 1, solve the model under sequential mode
        if max_threads == 1:
            solver.optimize()
        # Otherwise, solve the model under multi-threaded mode
        else:
            # Cap max_threads to 64 if it exceeds due to SCIP limitations
            max_threads = min(max_threads, 64)
            # Enable multi-threading to use all available cores
            solver.setParam("parallel/maxnthreads", max_threads)
            # Solve the model under concurrent mode
            solver.solveConcurrent()
        
        # Get the best solution
        sol = solver.getBestSol()

        # Print the solution to the output file if a path is provided
        if solution_file is not None:
            solver.writeBestSol(solution_file)

    # Create the new assignment starting with the original assignment
    assignment = original_assignment.assignment.copy()
    counts = [0 for _ in range(P)]
    # Update the new assignment with the solution
    for sample in range(workload.samples):
        rank = original_assignment.assignment['KppRank'][sample]
        # Check if the column is assigned to the processor
        # Round the val since it sometimes deviates a little bit from 1
        if solver.getSolVal(sol, x[counts[rank], rank]) >= 0.5:
            # Figure out which processor the column is assigned to
            for j in range(P):
                # Skip if j == rank
                if j == rank:
                    continue
                # Check if the column is assigned to the processor
                if solver.getSolVal(sol, s[rank, j]) == 1:
                    # Assign the column to the processor
                    assignment.loc[sample, 'KppRank'] = j
                    break
        # Otherwise, just ignore it and keep the original assignment
        # Increment the count of the processor regardless
        counts[rank] += 1
    # Change the name of KppRank to interval
    assignment.rename(columns={'KppRank': interval}, inplace=True)
    
    # Return the new assignment
    return Assignment(assignment)

# If ran as main, test the MIQCP function
if __name__ == "__main__":
    # Test the MIQCP function at c24 resolution at 6 processors
    res = 24
    workload = Workload.read_csv(f"test/workloads/c{res}.csv")
    procs = 24
    original_assignment = Assignment.read_csv(f"test/og_assignments/c{res}_p{procs}.csv")
    os.makedirs(f"test/MIQCP/c{res}_p{procs}", exist_ok=True)
    assignment = MIQCP(workload, original_assignment, 0, os.cpu_count(), f"test/MIQCP/c{res}_p{procs}/solution.txt")
    assignment.write_csv(f"test/MIQCP/c{res}_p{procs}/assignment.csv")
