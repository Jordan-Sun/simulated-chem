"""
run_greedy_interval.py
Runs the greedy heuristic for a range of intervals and writes the results to CSV files.
Usage: python run_greedy_interval.py <start_interval> <batch_size> <res> <hosts> <ptile> <swap_alg_name> <workload_base> <original_assignment_base> [seed] [mod]
"""
import sys
import os
from greedy import *
from workload import Workload
from assignment import Assignment

if len(sys.argv) < 9:
    print(sys.argv)
    print(
        "Usage: python run_greedy_interval.py <start_interval> <batch_size> <res> <hosts> <ptile> <swap_alg_name> <workload_base> <original_assignment_base> [seed] [mod]")
    sys.exit(1)

start_interval = int(sys.argv[1])
batch_size = int(sys.argv[2])
res = int(sys.argv[3])
hosts = int(sys.argv[4])
ptile = int(sys.argv[5])
swap_alg_name = sys.argv[6]
workload_base = sys.argv[7]
original_assignment_base = sys.argv[8]
seed = int(sys.argv[9]) if len(sys.argv) >= 10 else 0
mod = sys.argv[10] if len(sys.argv) >= 11 else ""
threshold_factor = 0.01
procs = hosts * ptile

# Compute end_interval based on batch_size and a reasonable upper bound
end_interval = start_interval + batch_size

if end_interval < start_interval:
    print(f"Skipping: end_interval ({end_interval}) < start_interval ({start_interval})")
    sys.exit(0)

if swap_alg_name == "greedy":
    swap_alg = greedy_swap
elif swap_alg_name == "dp":
    swap_alg = dp_swap
else:
    print("Invalid swap algorithm")
    sys.exit(1)

if mod:
    workload = Workload.read_csv(f"{workload_base}/{mod}c24_to_c{res}.csv")  # type: ignore
else:
    workload = Workload.read_csv(f"{workload_base}/c{res}.csv")

original_assignment = Assignment.read_csv(f"{original_assignment_base}/c{res}_p{procs}.csv")  # type: ignore

if hosts > 1:
    original_assignment.set_processor_groups(hosts, ptile, shuffle=seed)

base = f"test/{mod}{swap_alg_name}/c{res}_p{procs}"
if hosts > 1:
    base += f"_h{hosts}"
    if seed:
        base += f"_s{seed}"

os.makedirs(base, exist_ok=True)
os.makedirs(f"{base}/intervals", exist_ok=True)

# Cap the end_interval if workload has fewer intervals
if end_interval > workload.intervals:
    end_interval = workload.intervals
    print(f"Adjusted end_interval to {end_interval} based on workload size.")

for interval in range(start_interval, end_interval):
    result_path = f'{base}/intervals/interval_{interval}.csv'
    if hosts == 1:
        greed_heuristic(
            workload, original_assignment, interval, swap_alg, result_path, threshold_factor=threshold_factor,
        )
    else:
        greed_heuristic_local(
            workload, original_assignment, interval, swap_alg, result_path, threshold_factor=threshold_factor,
        )
    print(f"Processed interval {interval} -> {result_path}")
