"""
run_greedy_interval.py
Runs the greedy heuristic for a single interval and writes the result to a CSV file.
Usage: python run_greedy_interval.py <interval> <res> <hosts> <ptile> <swap_alg_name> <workload_base> <original_assignment_base> <mod>
"""
import sys
import os
from pathlib import Path
from greedy import greed_heuristic, greedy_swap, dp_swap
from workload import Workload
from assignment import Assignment

if len(sys.argv) != 9:
    print("Usage: python run_greedy_interval.py <interval> <res> <hosts> <ptile> <swap_alg_name> <workload_base> <original_assignment_base> <mod>")
    sys.exit(1)

interval = int(sys.argv[1])
res = int(sys.argv[2])
hosts = int(sys.argv[3])
ptile = int(sys.argv[4])
swap_alg_name = sys.argv[5]
workload_base = sys.argv[6]
original_assignment_base = sys.argv[7]
mod = sys.argv[8]
procs = hosts * ptile

if swap_alg_name == "greedy":
    swap_alg = greedy_swap
elif swap_alg_name == "dp":
    swap_alg = dp_swap
else:
    print("Invalid swap algorithm")
    sys.exit(1)

if mod == "":
    workload = Workload.read_csv(Path(f"{workload_base}/c{res}.csv"))
else:
    workload = Workload.read_csv(Path(f"{workload_base}/{mod}c24_to_c{res}.csv"))

original_assignment = Assignment.read_csv(Path(f"{original_assignment_base}/c{res}_p{procs}.csv"))
base = f"test/{mod}{swap_alg_name}/c{res}_p{procs}"
os.makedirs(base, exist_ok=True)
os.makedirs(f"{base}/intervals", exist_ok=True)

result_path = f'{base}/intervals/interval_{interval}.csv'
greed_heuristic(
    workload, original_assignment, interval, swap_alg, result_path
)
print(f"Processed interval {interval} -> {result_path}")
