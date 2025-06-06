"""
combine_assignments.py
Combines all interval assignment CSVs into a single assignment file.
Usage: python combine_assignments.py <num_intervals> <res> <hosts> <ptile> <swap_alg_name> [mod]
"""
import sys
import os
from assignment import Assignment
from pathlib import Path

if len(sys.argv) < 7:
    print("Usage: python combine_assignments.py <num_intervals> <res> <hosts> <ptile> <swap_alg_name>  [mod]")
    sys.exit(1)

num_intervals = int(sys.argv[1])
res = int(sys.argv[2])
hosts = int(sys.argv[3])
ptile = int(sys.argv[4])
swap_alg_name = sys.argv[5]
mod = sys.argv[6] if len(sys.argv) >= 7 else ""
procs = hosts * ptile
base = f"test/{mod}{swap_alg_name}/c{res}_p{procs}"

assignments = []
def to_os_path(path):
    import os
    return os.fspath(path)
for interval in range(num_intervals):
    path = f"{base}/intervals/interval_{interval}.csv"
    assignments.append(Assignment.read_csv(to_os_path(path)))

assignment = Assignment.concatenate(assignments)
assignment.write_csv(to_os_path(f'{base}/assignment.csv'))
print(f"Combined {num_intervals} intervals into {base}/assignment.csv")
