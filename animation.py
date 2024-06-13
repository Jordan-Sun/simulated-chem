from typing import Tuple
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
    
    # Create the plot
    fig, ax = plt.subplots()
    ax.set_title('Computation Costs')
    ax.set_xlabel('Processor')
    ax.set_ylabel('Computation Cost')
    ax.set_xlim(0, processors)

    # Update the animation with the current frame / interval
    def update(interval: int):
        # Find the processor with the maximum work.
        intervalComputations = [0] * processors
        # Process each sample
        for x in range(width):
            for y in range(height):
                sample = whtoi(x, y, width, height)
                # Add the work to the total for the processor
                intervalComputations[assignment[sample]] += matrix[sample][interval]
        # Plot the computation costs as a bar chart
        ax.clear()
        ax.bar(range(processors), intervalComputations, color='blue')

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=intervals, repeat=False)

    # Return the animation
    return ani

