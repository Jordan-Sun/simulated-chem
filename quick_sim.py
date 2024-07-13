import numpy as np
import pandas as pd

# number of processors
processors = 24

# open the workload file
workload_file = 'nc4/workload.csv'
workload = pd.read_csv(workload_file, header='infer', index_col=0)

# infer the number of samples and intervals
samples, intervals = workload.shape

assignment_dir = 'nc4/p_24/dynamic_100'

# total cost and processor costs
total_cost = 0
processor_costs = np.zeros(processors)

# output csv header
print('Interval, Processor, Cost')

# iterate over all the intervals
for i in range(intervals):
    # reset the processor costs
    processor_costs.fill(0)
    
    # open the assignment file
    assignment_file = f'{assignment_dir}/interval_{i}.assignment'
    assignment = pd.read_csv(assignment_file, header=None).values.flatten()

    # tally the cost of each processor at this interval according to the assignment
    for n in range(samples):
        processor_costs[assignment[n]] += workload.iloc[n, i]

    # print the processor with the maximum cost
    print(f'{i}, {np.argmax(processor_costs)}, {processor_costs.max()}')
    # add the maximum cost to the total cost
    total_cost += processor_costs.max()

# print the total cost
print(f'Total, NA, {total_cost}')
