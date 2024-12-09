import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import Tuple

# Simulate the computation cost and plot it as an animation
def animate(matrix: list, assignment_file: str, samples: int, intervals: int, processors: int) -> animation.FuncAnimation:
    # Validate the input
    if samples != len(matrix):
        raise ValueError('Invalid number of samples, expected ' +
                         str(len(matrix)) + ', got ' + str(samples) + '.')
    if intervals != len(matrix[0]):
        raise ValueError('Invalid number of intervals, expected ' +
                         str(len(matrix[0])) + ', got ' + str(intervals) + '.')

    # Create a panda data frame to store the computation costs for each processor at each interval
    df = pd.DataFrame(index=range(intervals), columns=range(processors))
    # Find the maximum computation cost as well as the lower bound as we are at it
    maxCost = 0
    total = 0
    span = 0

    # Precopmute the computation costs for each processor at each interval
    for interval in range(intervals):
        assignment_file = f'{assignment_dir}/interval_{i}.assignment'
        assignment = pd.read_csv(assignment_file, header=None).values.flatten()
        intervalComputations = [0] * processors  
        for sample in range(samples):
            intervalComputations[assignment[sample]] += matrix[sample][interval]
            total += matrix[sample][interval]
        df.loc[interval] = intervalComputations
        localMax = max(intervalComputations)
        span += localMax
        maxCost = max(maxCost, localMax)

    # Compute the lower bound
    bound = min(total / processors, span)
    # Round the max cost up to its significant digits
    maxCost = 100000

    # Create the plot
    fig, ax = plt.subplots()
    ax.set_title('Computation Costs')
    ax.set_xlabel('Processor')
    ax.set_ylabel('Computation Cost')
    ax.set_xticks(range(processors))
    ax.set_ylim(0, maxCost)
    # Make the plot wider
    fig.set_size_inches(16, 6)

    # Create an empty bar plot to be updated
    plot = ax.bar(range(processors), [0] * processors)
    # @todo: compute the bound and add it to the plot, currently not working with animation

    # Update the animation with the current frame / interval
    def update(interval: int):
        # Update the bar plot with the current frame / interval
        for i, bar in enumerate(plot):
            bar.set_height(df.loc[interval][i])
        # Add the bound to the plot as a horizontal line
        ax.axhline(y=bound, color='r', linestyle='--')

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=intervals, repeat=False)

    # Return the animation
    return ani

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

# create the animation
ani = animate(workload.values, assignment_dir, samples, intervals, processors)
ani.save(assignment_dir + '/animation.gif', fps=1)
