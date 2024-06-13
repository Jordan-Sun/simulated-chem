from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Convert a 2d index to a 1d index with wrapping
def whtoi(x: int, y: int, width: int, height: int) -> int:
    if x < 0:
        x += width
    if x >= width:
        x -= width
    if y < 0:
        y += height
    if y >= height:
        y -= height
    return y * width + x

# Convert a 1d index to a 2d index
def itowh(i: int, width: int, height: int) -> Tuple[int, int]:
    if i < 0 or i >= width * height:
        raise ValueError('i out of bounds')
    return i % width, int(i / width)

# Simulate the computation cost and plot it as an animation
def animate(matrix: list, assignment: list, width: int, height: int, intervals: int, processors: int) -> animation.FuncAnimation:
    # Validate the input
    samples = width * height
    if samples != len(matrix):
        raise ValueError('Invalid number of samples, expected ' + str(len(matrix)) + ', got ' + str(samples) + '.')
    if intervals != len(matrix[0]):
        raise ValueError('Invalid number of intervals, expected ' + str(len(matrix[0])) + ', got ' + str(intervals) + '.')

    # Create a panda data frame to store the computation costs for each processor at each interval
    df = pd.DataFrame(index=range(intervals), columns=range(processors))
    # Find the maximum computation cost as well as the lower bound as we are at it
    maxCost = 0
    total = 0
    span = 0

    # Precopmute the computation costs for each processor at each interval
    for interval in range(intervals):
        intervalComputations = [0] * processors
        for x in range(width):
            for y in range(height):
                sample = whtoi(x, y, width, height)
                intervalComputations[assignment[sample]] += matrix[sample][interval]
                total += matrix[sample][interval]
        df.loc[interval] = intervalComputations
        localMax = max(intervalComputations)
        span += localMax
        maxCost = max(maxCost, localMax)

    # Round the max cost up to its significant digits
    roundUnit = 10 ** (len(str(int(maxCost))) - 1)
    maxCost = (maxCost // roundUnit + 1) * roundUnit

    # Compute the lower bound
    bound = min(total / processors, span)

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

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=intervals, repeat=False)

    # Return the animation
    return ani

