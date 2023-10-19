'''
complex_assignment.py

This file contains the functions that will be used to assign the samples to the processors.
Each funtion takes in the width x and height y of the sample grid, the number of intervals t, the number of processors, and workload matrix.
Each cell in the list represents the processor that the sample is assigned to.
'''
import random
import copy
import numpy as np
from typing import List, Tuple
    
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

def get_neighbor(sample: int, width: int, height: int) -> List[int]:
    x, y = itowh(sample, width, height)
    left = whtoi(x - 1, y, width, height)
    right = whtoi(x + 1, y, width, height)
    up = whtoi(x, y - 1, width, height)
    down = whtoi(x, y + 1, width, height)
    return [left, right, up, down] 

# Computation Only
def greedy(matrix: list, width: int, height: int, intervals: int, processors: int, extra: int) -> List[int]:
    # the total number of samples is width times height
    samples = width * height
    # the list of costs of processors
    processor_costs = []
    for _ in range(processors):
        processor_costs.append([0] * intervals)
    # the list of assignments
    assignments = [0] * samples 
    # the costs as of last iteration
    last_costs = [0] * intervals

    # preprocess sample order from large to small
    # samples_list = sorted(range(len(matrix)), key=lambda i: sum(matrix[i]), reverse=True)
    # preprocess sample order randomly
    samples_list = list(range(samples))
    if extra != -1:
        random.seed(extra)
        random.shuffle(samples_list)

    # greedily loop through all the samples
    for sample in samples_list:
        # compute the new cost of assigning the sample to each processor
        costs = [0] * processors
        for processor in range(processors):
            # print(processor_costs[processor])
            for i in range(intervals):
                # the new cost of this processor at this interval
                cost = processor_costs[processor][i] + matrix[sample][i]
                # the cost of this interval is the max of the last cost and the new cost
                costs[processor] += max(last_costs[i], cost)
        # the processor with the first lowest cost is the one we assign the sample to
        min_processor = costs.index(min(costs)) # return the leftmost occurance
        # print('Sample {} -> Processor {}'.format(sample, min_processor))
        # update the assignments
        assignments[sample] = min_processor
        # update the processor costs and last costs
        for i in range(intervals):
            processor_costs[min_processor][i] += matrix[sample][i]
            last_costs[i] = max(last_costs[i], processor_costs[min_processor][i])

    return assignments

# partial assignment is the list of assignments so far
# next assignment sample is the sample that will be assigned next
# next assignment proc is the processor that the sample will be assigned to
# processor costs is the list of costs of each processor at each interval so far
# already broadcasted is the list of samples that have already been broadcasted
def independent_broadcast_cost_function(matrix: list, partial_assignment: List[int], next_assignment_sample: int, next_assignment_proc: int, processor_costs: List[List[int]], communication_costs: List[int], already_broadcasted: List[int], neighbors: List[int], secondary_neighbors: List[int], intervals: int, processors: int, send_cost: float, recv_cost: float):
    # compute the cost of assigning the sample to the processor
    for i in range(intervals):
        # the new cost of this processor at this interval
        processor_costs[next_assignment_proc][i] += matrix[next_assignment_sample][i]
    should_broadcast = False
    for neighbor in neighbors:
        neighbor_proc = partial_assignment[neighbor]
        # if the neighbor is not assigned to the same processor
        if neighbor_proc != next_assignment_proc and neighbor_proc != -1:
            should_broadcast = True
            if already_broadcasted[neighbor] == 0:
                # add the send cost to the neighbor's processor
                already_broadcasted[neighbor] = 1
                communication_costs[neighbor_proc] += send_cost
            communication_costs[next_assignment_proc] += recv_cost
            communication_costs[neighbor_proc] += recv_cost
    if should_broadcast:
        # add the send cost to the processor
        already_broadcasted[next_assignment_sample] = 1
        communication_costs[next_assignment_proc] += send_cost
    
    # compute the total cost
    total_cost = 0
    # processor cost for each interval
    for i in range(intervals):
        max_cost = 0
        for processor in range(processors):
            if processor_costs[processor][i] > max_cost:
                max_cost = processor_costs[processor][i]
        total_cost += max_cost
    # communication cost
    max_cost = 0
    for processor in range(processors):
        if communication_costs[processor] > max_cost:
            max_cost = communication_costs[processor]
    total_cost += max_cost * intervals
    return total_cost

def independent_unicast_cost_function(matrix: list, partial_assignment: List[int], next_assignment_sample: int, next_assignment_proc: int, processor_costs: List[List[int]], communication_costs: List[int], _: List[int], neighbors: List[int], secondary_neighbors: List[int], intervals: int, processors: int, send_cost: float, recv_cost: float):
    # compute the cost of assigning the sample to the processor
    for i in range(intervals):
        # the new cost of this processor at this interval
        processor_costs[next_assignment_proc][i] += matrix[next_assignment_sample][i]
    for neighbor in neighbors:
        neighbor_proc = partial_assignment[neighbor]
        # if the neighbor is not assigned to the same processor
        if neighbor_proc != next_assignment_proc and neighbor_proc != -1:
            communication_costs[next_assignment_proc] += send_cost
            communication_costs[neighbor_proc] += send_cost
            communication_costs[next_assignment_proc] += recv_cost
            communication_costs[neighbor_proc] += recv_cost
    
    # compute the total cost
    total_cost = 0
    # processor cost for each interval
    for i in range(intervals):
        max_cost = 0
        for processor in range(processors):
            if processor_costs[processor][i] > max_cost:
                max_cost = processor_costs[processor][i]
        total_cost += max_cost
    # communication cost
    max_cost = 0
    for processor in range(processors):
        if communication_costs[processor] > max_cost:
            max_cost = communication_costs[processor]
    total_cost += max_cost * intervals
    return total_cost

def dependent_broadcast_cost_function(matrix: list, partial_assignment: List[int], next_assignment_sample: int, next_assignment_proc: int, processor_costs: List[List[int]], communication_costs: List[int], already_broadcasted: List[int], neighbors: List[int], secondary_neighbors: List[int], intervals: int, processors: int, send_cost: float, recv_cost: float):
    # compute the cost of assigning the sample to the processor
    for i in range(intervals):
        # the new cost of this processor at this interval
        processor_costs[next_assignment_proc][i] += matrix[next_assignment_sample][i]
    should_broadcast = False
    for neighbor in neighbors:
        neighbor_proc = partial_assignment[neighbor]
        # if the neighbor is not assigned to the same processor
        if neighbor_proc != next_assignment_proc and neighbor_proc != -1:
            should_broadcast = True
            if already_broadcasted[neighbor] == 0:
                # add the send cost to the neighbor's processor
                already_broadcasted[neighbor] = 1
                communication_costs[neighbor_proc] += send_cost
            communication_costs[next_assignment_proc] += recv_cost
            communication_costs[neighbor_proc] += recv_cost
    if should_broadcast:
        # add the send cost to the processor
        already_broadcasted[next_assignment_sample] = 1
        communication_costs[next_assignment_proc] += send_cost
    
    # compute the total cost
    total_cost = 0
    # processor cost for each interval
    for i in range(intervals):
        max_cost = 0
        for processor in range(processors):
            if processor_costs[processor][i] + communication_costs[processor] > max_cost:
                max_cost = processor_costs[processor][i] + communication_costs[processor]
        total_cost += max_cost
    return total_cost

def dependent_unicast_cost_function(matrix: list, partial_assignment: List[int], next_assignment_sample: int, next_assignment_proc: int, processor_costs: List[List[int]], communication_costs: List[int], _: List[int], neighbors: List[int], secondary_neighbors: List[int], intervals: int, processors: int, send_cost: float, recv_cost: float):
    # compute the cost of assigning the sample to the processor
    for i in range(intervals):
        # the new cost of this processor at this interval
        processor_costs[next_assignment_proc][i] += matrix[next_assignment_sample][i]
    for neighbor in neighbors:
        neighbor_proc = partial_assignment[neighbor]
        # if the neighbor is not assigned to the same processor
        if neighbor_proc != next_assignment_proc and neighbor_proc != -1:
            communication_costs[next_assignment_proc] += send_cost
            communication_costs[neighbor_proc] += send_cost
            communication_costs[next_assignment_proc] += recv_cost
            communication_costs[neighbor_proc] += recv_cost
    
    # compute the total cost
    total_cost = 0
    # processor cost for each interval
    for i in range(intervals):
        max_cost = 0
        for processor in range(processors):
            if processor_costs[processor][i] + communication_costs[processor] > max_cost:
                max_cost = processor_costs[processor][i] + communication_costs[processor]
        total_cost += max_cost
    return total_cost

def weak_neighbor_dependent_unicast_cost_function(matrix: list, partial_assignment: List[int], next_assignment_sample: int, next_assignment_proc: int, processor_costs: List[List[int]], communication_costs: List[int], _: List[int], neighbors: List[int], secondary_neighbors: List[int], intervals: int, processors: int, send_cost: float, recv_cost: float):
    # compute the cost of assigning the sample to the processor
    for i in range(intervals):
        # the new cost of this processor at this interval
        processor_costs[next_assignment_proc][i] += matrix[next_assignment_sample][i]
    for neighbor in neighbors:
        neighbor_proc = partial_assignment[neighbor]
        # if the neighbor is not assigned to the same processor
        if neighbor_proc != next_assignment_proc and neighbor_proc != -1:
            communication_costs[next_assignment_proc] += send_cost
            communication_costs[neighbor_proc] += send_cost
            communication_costs[next_assignment_proc] += recv_cost
            communication_costs[neighbor_proc] += recv_cost
    secondary_factor = 0.3
    for neighbor in secondary_neighbors:
        neighbor_proc = partial_assignment[neighbor]
        # if the neighbor is not assigned to the same processor
        if neighbor_proc != next_assignment_proc and neighbor_proc != -1:
            communication_costs[next_assignment_proc] += secondary_factor * send_cost
            communication_costs[neighbor_proc] += secondary_factor * send_cost
            communication_costs[next_assignment_proc] += secondary_factor * recv_cost
            communication_costs[neighbor_proc] += secondary_factor * recv_cost
    
    # compute the total cost
    total_cost = 0
    # processor cost for each interval
    for i in range(intervals):
        max_cost = 0
        for processor in range(processors):
            if processor_costs[processor][i] + communication_costs[processor] > max_cost:
                max_cost = processor_costs[processor][i] + communication_costs[processor]
        total_cost += max_cost
    return total_cost

def greedy_with_communication(matrix: list, width: int, height: int, intervals: int, processors: int, extra: int, send_cost: float, recv_cost: float, cost_function) -> List[int]:
    # the total number of samples is width times height
    samples = width * height
    # the list of assignments, -1 means unassigned
    assignments = [-1] * samples
    # the list of costs of processors
    processor_costs = []
    for _ in range(processors):
        processor_costs.append([0] * intervals)
    # the list of communication costs of processors
    communication_costs = [0] * processors
    # if the sample has broadcasted
    already_broadcasted = [0] * samples

    # preprocess data based on total cost in diminishing order
    # samples_list = sorted(range(len(matrix)), key=lambda i: sum(matrix[i]), reverse=True)
    samples_list = list(range(samples))
    if extra != -1:
        random.seed(extra)
        random.shuffle(samples_list)
    # greedily loop through all the samples
    for sample in samples_list:
        # the coordinates of the sample
        x, y = itowh(sample, width, height)
        # first level neighbors
        # left
        left = whtoi(x - 1, y, width, height)
        # right
        right = whtoi(x + 1, y, width, height)
        # up
        up = whtoi(x, y - 1, width, height)
        # down
        down = whtoi(x, y + 1, width, height)
        # the list of neighbors
        neighbors = [left, right, up, down]
        # second level neighbors
        left_left = whtoi(x - 2, y, width, height)
        left_up = whtoi(x - 1, y - 1, width, height)
        up_up = whtoi(x, y - 2, width, height)
        right_up = whtoi(x + 1, y - 1, width, height)
        right_right = whtoi(x + 2, y, width, height)
        right_down = whtoi(x + 1, y + 1, width, height)
        down_down = whtoi(x, y + 2, width, height)
        left_down = whtoi(x - 1, y + 1, width, height)
        # the list of secondary neighbors
        secondary_neighbors = [left_left, left_up, up_up, right_up, right_right, right_down, down_down, left_down]
        # compute the new total cost of assigning the sample to each processor by calling the cost function
        min_cost = float('inf')
        min_processor = -1
        min_processor_costs = []
        min_communication_costs = []
        min_already_broadcasted = []
        for processor in range(processors):
            temp_processor_costs = copy.deepcopy(processor_costs)
            temp_communication_costs = copy.deepcopy(communication_costs)
            temp_already_broadcasted = copy.deepcopy(already_broadcasted)
            # the cost of assigning this sample to this processor
            temp_cost = cost_function(matrix, assignments, sample, processor, temp_processor_costs, temp_communication_costs, temp_already_broadcasted, neighbors, secondary_neighbors, intervals, processors, send_cost, recv_cost)
            # if this cost is lower than the current min cost, update the min cost and min processor
            if temp_cost < min_cost:
                min_cost = temp_cost
                min_processor = processor
                min_processor_costs = temp_processor_costs
                min_communication_costs = temp_communication_costs
                min_already_broadcasted = temp_already_broadcasted
        # assign the sample to the processor with the lowest cost
        assignments[sample] = min_processor
        processor_costs = min_processor_costs
        communication_costs = min_communication_costs
        already_broadcasted = min_already_broadcasted
    
    return assignments

def greedy_prioritize_communication(matrix: list, width: int, height: int, intervals: int, processors: int, extra: int, send_cost: float, recv_cost: float, constant: int) -> List[int]:
    # print("Here is matrix: \n", matrix)
    samples = height * width
    assignment = [-1] * samples
    processor_workload_array = np.zeros(processors * intervals).reshape(processors, intervals)
    workload_array = np.array(matrix)
    # print("workload_array: ",workload_array)
    # print("workload_array size: ",workload_array.shape)
    # print("height=",height)
    # print("width=",width)
    # print("interval=",intervals)
    # print("samples=",samples)
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if matrix else 0
    computation_overload_threshold = constant * np.sum(workload_array, axis=0).reshape(-1, 1)/processors
    # print("computation_overload_threshold\n", computation_overload_threshold)
    samples_list = list(range(samples))
    if extra != -1:
        random.seed(extra)
        random.shuffle(samples_list)
    random_choice = 0
    for sample in samples_list:
        min_processor = -1
        # the coordinates of the sample
        x, y = itowh(sample, width, height)
        # first level neighbors
        # left
        left = whtoi(x - 1, y, width, height)
        # right
        right = whtoi(x + 1, y, width, height)
        # up
        up = whtoi(x, y - 1, width, height)
        # down
        down = whtoi(x, y + 1, width, height)
        # the list of neighbors
        neighbors = [left, right, up, down]
        arr = np.zeros(processors)
        for neighbor in neighbors:
            if assignment[neighbor] != -1:
                arr[assignment[neighbor]] += 1
        dictionary = dict(zip(list(range(processors)), arr))
        dictionary = {k: v for k, v in dictionary.items() if v != 0}
        dictionary = dict(sorted(dictionary.items(), key = lambda item: item[1], reverse = True))
        ranked_array = list(dictionary.keys())
        for ranked_processor in ranked_array:
            tmp_array = processor_workload_array[ranked_processor].reshape(-1, 1) + workload_array[sample].reshape(-1, 1)
            is_overloaded = False
            for index in range(len(tmp_array)):
                if(tmp_array[index]>computation_overload_threshold[index]):
                    is_overloaded = True
                    break
            if is_overloaded:
                continue
            min_processor = ranked_processor
            break
        if min_processor != -1:
            assignment[sample] = min_processor
        else: 
            min_cost = float("inf")
            for processor in range(processors):
                tmp_processor_workload_array = copy.deepcopy(processor_workload_array)
                tmp_processor_workload_array[processor] += workload_array[sample]
                tmp_cost = np.sum(np.max(tmp_processor_workload_array, axis = 0))
                if min_cost > tmp_cost:
                    min_processor = processor
                    min_cost = tmp_cost
            random_choice += 1
            assignment[sample] = min_processor
            # print("min_processor: ",min_processor)
            # print("assignment\n",assignment)
        # print("processor_workload_array\n",processor_workload_array)
        processor_workload_array[min_processor] += workload_array[sample]
    # print("processor_workload_array",processor_workload_array)
    print("computation_overload_threshold\n", computation_overload_threshold)
    print("greedy choice percentage: ",random_choice/samples)
    # print("Number of greedy assignment: ",random_choice)
    return assignment

# Start by greedily assigning the sample to a processor.
# Repeatedly try to pack the block one larger that can be greedily packed to a processor under the threshold.
# Assign it to largest processor that can fit it.
def mosaic_greedy(matrix: list, width: int, height: int, intervals: int, processors: int, extra: int, constant: int) -> List[int]:
    # the total number of samples is width times height
    samples = width * height
    # the list of assignments, -1 means unassigned
    assignments = [-1] * samples
    # the list of costs of processors
    processor_costs = []
    for _ in range(processors):
        processor_costs.append([0] * intervals)
    # the max costs at each interval as of last iteration
    last_costs = [0] * intervals

    # compute the threshold for each interval
    threshold = [0] * intervals
    for i in range(intervals):
        threshold[i] = constant * sum(matrix[sample][i] for sample in range(samples)) / processors

    # shuffle the processor if extra is not -1
    samples_list = list(range(samples))
    if extra != -1:
        random.seed(extra)
        random.shuffle(samples_list)

    # count of each block size
    block_sizes = {}

    # loop through all the samples
    for sample in samples_list:
        # check if the sample has been previously assigned to a processor in pervious iterations
        if assignments[sample] != -1:
            continue

        # debug print
        # print('Processing sample {}'.format(sample))
        
        # greedily choose the initial processor
        # the block size that can be greedily packed to a processor
        tentative_block_size = 1
        # the samples in the block
        tentative_block_samples = [sample]
        # compute the new cost of assigning the sample to each processor
        costs = [0] * processors
        for processor in range(processors):
            for i in range(intervals):
                # the new cost of this processor at this interval
                cost = processor_costs[processor][i] + matrix[sample][i]
                # the cost of this interval is the max of the last cost and the new cost
                costs[processor] += max(last_costs[i], cost)
        # assign to the leftmost processor with the lowest cost
        tentative_block_processor = costs.index(min(costs))
        
        # the coordinates of the sample
        x, y = itowh(sample, width, height)
        # greedily pack larger block to the processor until it reaches the threshold
        while True:
            # the new block size
            block_size = tentative_block_size + 1
            # unasigned samples in the block
            block_samples = []
            # cost of the block at each interval
            block_costs = [0] * intervals
            for j in range(y, y + block_size):
                # check if the height is out of bound
                if j >= height:
                    break
                for i in range(x, x + block_size):
                    # check if the width is out of bound
                    if i >= width:
                        break
                    # skip if the sample is already assigned
                    if assignments[whtoi(i, j, width, height)] != -1:
                        continue
                    # add the sample to the block
                    block_samples.append(whtoi(i, j, width, height))
                    # add the cost of the sample to the block cost
                    for k in range(intervals):
                        block_costs[k] += matrix[whtoi(i, j, width, height)][k]

            # compute the new cost of assigning the block to each processor
            min_cost = float('inf')
            min_processor = -1
            for processor in range(processors):
                valid = True
                cost = 0
                interval_costs = [0] * intervals
                for i in range(intervals):
                    # the new cost of this processor at this interval
                    interval_costs[i] = processor_costs[processor][i] + block_costs[i]
                    # check if the cost at the interval is over the threshold
                    if interval_costs[i] > threshold[i]:
                        # if the cost is over the threshold, the block is not valid
                        valid = False
                        break
                    # the cost of this interval is the max of the last cost and the new cost
                    cost += max(last_costs[i], interval_costs[i])
                # if the block is valid
                if valid:
                    # if the cost is less than the current min cost, update the min cos, min interval costs, and min processor
                    if cost < min_cost:
                        min_cost = cost
                        min_processor = processor
            
            # if min processor is not -1, the block can be assigned to a processor without exceeding the threshold
            if min_processor != -1:
                # update the tentative assignments
                tentative_block_size = block_size
                tentative_block_samples = block_samples
                tentative_block_processor = min_processor
            # if the block cannot be assigned to a processor without exceeding the threshold
            else:
                break    

        # assign the tentative block to the tentative processor
        for sample in tentative_block_samples:
            assignments[sample] = tentative_block_processor
        # update the processor costs and last costs
        for i in range(intervals):
            for sample in tentative_block_samples:
                processor_costs[tentative_block_processor][i] += matrix[sample][i]
            last_costs[i] = max(last_costs[i], processor_costs[tentative_block_processor][i])

        # update the block size count
        if tentative_block_size in block_sizes:
            block_sizes[tentative_block_size] += 1
        else:
            block_sizes[tentative_block_size] = 1
    
    # print the block size count statistics
    # print('Block Size Count:')
    # for block_size in block_sizes:
    #     print('{}: {}'.format(block_size, block_sizes[block_size]))

    return assignments