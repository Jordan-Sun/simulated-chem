from workload import Workload
from assignment import Assignment
import diff_proc

import sys
import os
import multiprocessing
import contextlib
from functools import partial

# Thread used to process each interval
def process_interval(interval: int, root: str, bias: float = 0.0, const: float = 0.0, mem: float = None) -> Assignment:
    print(f"Processing interval {interval}")
    # Redirect output to log/{interval}.txt
    with open(f"{root}/logs/{interval}.txt", 'w') as f:
        with contextlib.redirect_stdout(f):
            # Ask MIQCP to redirect output to python and let us redirect it
            assignment = diff_proc.reassign(
                workload, original_assignment, interval, 1, f"{root}/solutions/{interval}.txt", True, bias, const, mem)
    print(f"Finished interval {interval}")
    return assignment

# Read res and procs from arguments
if len(sys.argv) < 5:
    print("Usage: python main.py <res> <procs> <bias> <const> [interval]")
    sys.exit(1)
res = int(sys.argv[1])
procs = int(sys.argv[2])
bias = float(sys.argv[3])
const = float(sys.argv[4])
if len(sys.argv) == 6:
    interval = int(sys.argv[5])
else:
    interval = None
total_mem = 16000.0

# Read workload file
workload = Workload.read_csv(f"test/workloads/c{res}.csv")

# Read original assignment
original_assignment = Assignment.read_csv(
    f"test/og_assignments/c{res}_p{procs}.csv")

# Run dynamic constrained MIQCP for each interval in parallel
root = f"test/diff_MIQCP/c{res}_p{procs}"
if bias != 0.0:
    root = root + f"_b{bias}"
if const != 0.0:
    root = root + f"_c{const}"


os.makedirs(f"{root}/logs", exist_ok=True)
os.makedirs(f"{root}/solutions", exist_ok=True)

if interval is None:
    # Run all intervals in parallel
    proc_mem = total_mem / os.cpu_count()
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        assignments = pool.map(partial(process_interval, root=root, bias=bias, const=const, mem=proc_mem), range(workload.intervals), chunksize = 1)
    print("Generating assignment.csv")
    # Concatenate assignments and write to file
    assignment = Assignment.concatenate(assignments)
    assignment.write_csv(f"{root}/assignment.csv")
else:
    # Fix specific interval and ignore the assignment output
    # Output directly to stdout
    _ = diff_proc.reassign(
        workload, original_assignment, interval, os.cpu_count(), f"{root}/solutions/{interval}.txt", False, bias, const, total_mem)
