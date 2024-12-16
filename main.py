from workload import Workload
from assignment import Assignment
import dynamic_constrained

import sys
import os
import multiprocessing
import contextlib
from functools import partial

# Thread used to process each interval
def process_interval(interval: int, root: str, bias: float = 0.0) -> Assignment:
    print(f"Processing interval {interval}")
    # Redirect output to log/{interval}.txt
    with open(f"{root}/logs/{interval}.txt", 'w') as f:
        with contextlib.redirect_stdout(f):
            # Ask MIQCP to redirect output to python and let us redirect it
            assignment = dynamic_constrained.MIQCP(
                workload, original_assignment, interval, 1, f"{root}/solutions/{interval}.txt", True, bias)
    print(f"Finished interval {interval}")
    return assignment

# Read res and procs from arguments
if len(sys.argv) < 4:
    print("Usage: python main.py <res> <procs> <bias> [interval]")
    sys.exit(1)
res = int(sys.argv[1])
procs = int(sys.argv[2])
bias = float(sys.argv[3])
if len(sys.argv) == 5:
    interval = int(sys.argv[4])
else:
    interval = None

# Read workload file
workload = Workload.read_csv(f"test/workloads/c{res}.csv")

# Read original assignment
original_assignment = Assignment.read_csv(
    f"test/og_assignments/c{res}_p{procs}.csv")

# Run dynamic constrained MIQCP for each interval in parallel
if bias == 0.0:
    root = f"test/MIQCP/c{res}_p{procs}"
else:
    root = f"test/MIQCP/c{res}_p{procs}_b{bias}"

os.makedirs(f"{root}/logs", exist_ok=True)
os.makedirs(f"{root}/solutions", exist_ok=True)
if interval is None:
    # Run all intervals in parallel
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        assignments = pool.map(partial(process_interval, root=root, bias=bias), range(workload.intervals), chunksize=1)
    print("Generating assignment.csv")
    # Concatenate assignments and write to file
    assignment = Assignment.concatenate(assignments)
    assignment.write_csv(f"{root}/assignment.csv")
else:
    # Fix specific interval and ignore the assignment output
    # Output directly to stdout
    _ = dynamic_constrained.MIQCP(
        workload, original_assignment, interval, os.cpu_count(), f"{root}/solutions/{interval}.txt", False, bias)
