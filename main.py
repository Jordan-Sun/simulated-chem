from workload import Workload
from assignment import Assignment
import dynamic_constrained

import sys
import os
import multiprocessing
import contextlib

# Thread used to process each interval
def process_interval(interval: int) -> Assignment:
    print(f"Processing interval {interval}")
    # Redirect output to log_{interval}.txt
    with open(f"test/MIQCP/c{res}_p{procs}/logs/{interval}.txt", 'w') as f:
        with contextlib.redirect_stdout(f):
            # Ask MIQCP to redirect output to python and let us redirect it to log_{interval}.txt
            assignment = dynamic_constrained.MIQCP(
                workload, original_assignment, interval, 1, f"test/MIQCP/c{res}_p{procs}/solutions/{interval}.txt", True)
    return assignment

# Read res and procs from arguments
if len(sys.argv) != 3:
    print("Usage: python main.py <res> <procs>")
    sys.exit(1)
res = int(sys.argv[1])
procs = int(sys.argv[2])

# Read workload file
workload = Workload.read_csv(f"test/workloads/c{res}.csv")

# Read original assignment
original_assignment = Assignment.read_csv(
    f"test/og_assignments/c{res}_p{procs}.csv")

# Run dynamic constrained MIQCP for each interval in parallel
os.makedirs(f"test/MIQCP/c{res}_p{procs}/logs", exist_ok=True)
os.makedirs(f"test/MIQCP/c{res}_p{procs}/solutions", exist_ok=True)
with multiprocessing.Pool(processes=os.cpu_count()) as pool:
    assignments = pool.map(process_interval, range(workload.intervals))
# Concatenate assignments and write to file
assignment = Assignment.concatenate(assignments)
assignment.write_csv(f"test/MIQCP/c{res}_p{procs}/assignment.csv")
