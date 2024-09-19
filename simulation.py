from typing import Tuple

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

# Compute the workspan of a workload matrix
def workspan(matrix: list, samples: int, intervals: int) -> Tuple[int, int]:
    if samples != len(matrix):
        raise ValueError('Invalid number of samples, expected ' + str(len(matrix)) + ', got ' + str(samples) + '.')
    if intervals != len(matrix[0]):
        raise ValueError('Invalid number of intervals, expected ' + str(len(matrix[0])) + ', got ' + str(intervals) + '.')
    # total span can be calculated as the sum of the maximums of each row
    span = sum(max(matrix[sample][interval] for sample in range(samples)) for interval in range(intervals))
    # total work is the sum of all works
    work = sum(matrix[sample][interval] for sample in range(samples) for interval in range(intervals))
    
    return span, work

# Compute the lower bound for the time needed for the given number of processors
def bound(span: int, work: int, processors: int) -> int:
    bound = int(work / processors)
    if bound < span:
        bound = span
    return bound

# Simulate the computation and communication cost of an assignment on a matrix
def simulate(matrix: list, assignment: list, width: int, height: int, intervals: int, processors: int, sendCost: int, recvCost: int) -> Tuple[int, int, int, int, int]:
    # Validate the input
    samples = width * height
    if samples != len(matrix):
        raise ValueError('Invalid number of samples, expected ' + str(len(matrix)) + ', got ' + str(samples) + '.')
    if intervals != len(matrix[0]):
        raise ValueError('Invalid number of intervals, expected ' + str(len(matrix[0])) + ', got ' + str(intervals) + '.')
    
    # Compute the total costs
    totalComputation = 0
    totalBroadcast = 0
    totalUnicast = 0
    totalComputationBroadcast = 0
    totalComputationUnicast = 0

    # Process each interval
    for interval in range(intervals):
        # Find the processor with the maximum work.
        intervalComputations = [0] * processors
        intervalBroadcasts = [0] * processors
        intervalUnicasts = [0] * processors
        intervalComputationBroadcasts = [0] * processors
        intervalComputationUnicasts = [0] * processors
        # Process each sample
        for x in range(width):
            for y in range(height):
                sample = whtoi(x, y, width, height)
                # Find the number of neighbors with a different assignment
                different = 0
                # Check top
                if assignment[sample] != assignment[whtoi(x, y - 1, width, height)]:
                    different += 1
                # Check bottom
                if assignment[sample] != assignment[whtoi(x, y + 1, width, height)]:
                    different += 1
                # Check left
                if assignment[sample] != assignment[whtoi(x - 1, y, width, height)]:
                    different += 1
                # Check right
                if assignment[sample] != assignment[whtoi(x + 1, y, width, height)]:
                    different += 1
                # Add the work to the total for the processor
                intervalComputations[assignment[sample]] += matrix[sample][interval]
                intervalBroadcasts[assignment[sample]] += sendCost * min(1, different) + recvCost * different
                intervalUnicasts[assignment[sample]] += (sendCost + recvCost) * different
                intervalComputationBroadcasts[assignment[sample]] += sendCost * min(1, different) + recvCost * different + matrix[sample][interval]
                intervalComputationUnicasts[assignment[sample]] += (sendCost + recvCost) * different + matrix[sample][interval]
        # Print the computation at this interval for all processors
        print('Interval', interval, 'computation:', intervalComputations)
        # Add the maximum work to the total.
        totalComputation += max(intervalComputations)
        totalBroadcast += max(intervalBroadcasts)
        totalUnicast += max(intervalUnicasts)
        totalComputationBroadcast += max(intervalComputationBroadcasts)
        totalComputationUnicast += max(intervalComputationUnicasts)
    return totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast

# Simulate the computation and communication cost of an assignment on a matrix with alternative communcation calculation
def alt_simulate(matrix: list, assignment: list, width: int, height: int, intervals: int, processors: int, sendCost: int, recvCost: int) -> Tuple[int, int]:
    # Validate the input
    samples = width * height
    if samples != len(matrix):
        raise ValueError('Invalid number of samples, expected ' + str(len(matrix)) + ', got ' + str(samples) + '.')
    if intervals != len(matrix[0]):
        raise ValueError('Invalid number of intervals, expected ' + str(len(matrix[0])) + ', got ' + str(intervals) + '.')

    # Compute the total costs
    totalComputation = 0
    totalUnicast = 0

    # Constant variables (update these to be inputs later)
    neighbor_offsets = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            if x != 0 or y != 0:
                neighbor_offsets.append((x, y))

    # Process each interval
    for interval in range(intervals):
        # Find the processor with the maximum work.
        intervalComputations = [0] * processors
        intervalUnicasts = [0] * processors
        # Process each sample
        for x in range(width):
            for y in range(height):
                sample = whtoi(x, y, width, height)
                # Find the number of neighbors with a different assignment
                different = 0
                # Check neighbors
                for offset in neighbor_offsets:
                    neighbor = whtoi(x + offset[0], y + offset[1], width, height)
                    if assignment[sample] != assignment[neighbor]:
                        different += 1
                # Add the work to the total for the processor
                intervalComputations[assignment[sample]] += matrix[sample][interval]
                intervalUnicasts[assignment[sample]] += (sendCost + recvCost) * different
        # Add the maximum work to the total.
        totalComputation += max(intervalComputations)
        totalUnicast += max(intervalUnicasts)
    return totalComputation, totalUnicast