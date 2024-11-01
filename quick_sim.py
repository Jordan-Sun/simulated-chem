import numpy as np
import pandas as pd
import sys

# number of processors
processors = 24

# open the workload file
workload_file = 'nc4/workload.csv'
workload = pd.read_csv(workload_file, header='infer', index_col=0)

# infer the number of samples and intervals
samples, intervals = workload.shape

# use the first argument as the assignment directory
if len(sys.argv) != 2:
    print("Usage: python {} <assignment_dir>".format(sys.argv[0]))
    sys.exit(1)
assignment_dir = sys.argv[1]

# total cost and processor costs
total_cost = 0
total_bound = 0
processor_costs = np.zeros(processors)

# output csv header
print('Interval, Processor, Cost, Lower Bound, Delta')

# iterate over all the intervals
for i in range(intervals):
    # reset the processor costs
    processor_costs.fill(0)
    
    # open the assignment file
    assignment_file = f'{assignment_dir}/interval_{i}.assignment'
    assignment = pd.read_csv(assignment_file, header=None).values.flatten()
    lower_bound = 0
    max_work = 0

    # tally the cost of each processor at this interval according to the assignment
    for n in range(samples):
        work = workload.iloc[n, i]
        processor_costs[assignment[n]] += work
        lower_bound += work
        if work > max_work:
            max_work = work

    # print the processor with the maximum cost
    max_proc = np.argmax(processor_costs)
    max_cost = processor_costs[max_proc]
    lower_bound /= processors
    if max_work > lower_bound:
        lower_bound = max_work
    print(f'{i}, {max_proc}, {max_cost}, {lower_bound}, {max_cost - lower_bound}')
    # add the maximum cost to the total cost
    total_cost += max_cost
    total_bound += lower_bound

# print the total cost
print(f'Total, NA, {total_cost}, {total_bound}, {total_cost - total_bound}')
