import simulation
import numpy as np
import time
import csv
import complex_assignment
import sys
import math
import copy
import random
from typing import List, Tuple
from itertools import chain

# def greedy_prioritize_communication_single_entry(matrix: list, width: int, height: int, intervals: int, processors: int, extra: int, send_cost: float, recv_cost: float, constant: float, coarse: int) -> List[int]:
#     samples = height * width
#     assignment = [-1] * samples
#     processor_workload_array = np.zeros(processors * intervals).reshape(processors, intervals)
#     workload_array = np.array(matrix)
#     num_rows = len(matrix)
#     num_cols = len(matrix[0]) if matrix else 0
#     computation_overload_threshold = constant * np.sum(workload_array, axis=0).reshape(-1, 1)/processors
#     samples_list = list(range(samples))
#     if extra != -1:
#         random.seed(extra)
#         random.shuffle(samples_list)
#     random_choice = 0
#     while len(samples_list)>0: 
#         sample = samples_list[0]
#         print("sample list: ",samples_list)
#         min_processor = -1
#         # the coordinates of the sample
#         x, y = itowh(sample, width, height)
#         # first level neighbors
#         # left
#         left = whtoi(x - 1, y, width, height)
#         # right
#         right = whtoi(x + 1, y, width, height)
#         # up
#         up = whtoi(x, y - 1, width, height)
#         # down
#         down = whtoi(x, y + 1, width, height)
#         # the list of neighbors
#         neighbors = [left, right, up, down]
#         # the list of indices of blocks around sample
#         arr = np.zeros(processors)
#         for neighbor in neighbors:
#             if assignment[neighbor] != -1:
#                 arr[assignment[neighbor]] += 1
#         blocks = get_block_index(coarse, x, y, width, height)
#         print("selected blocks: ",blocks)
#         # select unassigned tasks only and coversion between numpy array and lists
#         # mask = np.isin(blocks, samples_list)
#         # blocks_array = np.array(blocks)
#         # blocks_filtered = blocks_array[mask]
#         # blocks_filtered_list = blocks_filtered.tolist()
#         # print("available blocks: ",blocks_filtered_list)
#         dictionary = dict(zip(list(range(processors)), arr))
#         dictionary = {k: v for k, v in dictionary.items() if v != -1}
#         dictionary = dict(sorted(dictionary.items(), key = lambda item: item[1], reverse = True))
#         ranked_array = list(dictionary.keys())
#         for ranked_processor in ranked_array:
#             # calculate workload for block assignment
#             # blocks_workload_array = np.zeros(processor_workload_array[ranked_processor].reshape(-1, 1).shape)
#             # for block in blocks_filtered_list:
#             #     blocks_workload_array += workload_array[block].reshape(-1,1)
#             tmp_array = processor_workload_array[ranked_processor].reshape(-1, 1) + workload_array[sample].reshape(-1, 1)
#             is_overloaded = False
#             for index in range(len(tmp_array)):
#                 if(tmp_array[index] > computation_overload_threshold[index]):
#                     is_overloaded = True
#                     print("Overload")
#                     break
#             if is_overloaded:
#                 continue
#             min_processor = ranked_processor
#             break
#         print("min_processor: ",min_processor)
#         # if min_processor is found    
#         if min_processor != -1:
#             for block in blocks_filtered_list:
#                 # assign block to min_processor
#                 assignment[block] = min_processor
#                 # remove block from sample_list after it beinga assigned
#                 samples_list = np.delete(samples_list, np.where(samples_list == block))
#             print(" block assignment: ",assignment)
#         else: 
#             min_cost = float("inf")
#             for processor in range(processors):
#                 tmp_processor_workload_array = copy.deepcopy(processor_workload_array)
#                 tmp_processor_workload_array[processor] += workload_array[sample]
#                 tmp_cost = np.sum(np.max(tmp_processor_workload_array, axis = 0))
#                 if min_cost > tmp_cost:
#                     min_processor = processor
#                     min_cost = tmp_cost
#             random_choice += 1
#             # assign sample to min_processor
#             assignment[sample] = min_processor
#             # remove sample from sample_list after it beinga assigned
#             samples_list = np.delete(samples_list, np.where(samples_list == sample))
#             # print("min_processor: ",min_processor)
#             print("greedy assignment: ",assignment)
#         # print("processor_workload_array\n",processor_workload_array)
#         processor_workload_array[min_processor] += workload_array[sample]
#     # print("processor_workload_array",processor_workload_array)
#     print("computation_overload_threshold\n", computation_overload_threshold)
#     print("greedy choice percentage: ",random_choice/samples)
#     # print("Number of greedy assignment: ",random_choice)
#     return assignment

def greedy_prioritize_communication_fine_tune(matrix: list, width: int, height: int, intervals: int, processors: int, extra: int, send_cost: float, recv_cost: float, constant: float, coarse: int, single_entry: bool) -> List[int]:
    samples = height * width
    assignment = [-1] * samples
    processor_workload_array = np.zeros(processors * intervals).reshape(processors, intervals)
    workload_array = np.array(matrix)
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if matrix else 0
    computation_overload_threshold = constant * np.sum(workload_array, axis=0).reshape(-1, 1)/processors
    coarsen_workload_array, coarse_width, coarse_height, coarse_dictionary = coarsen(workload_array, width, height, intervals, coarse)
    # print("coarse_dictionary: ",coarse_dictionary)
    # print("coarsen_workload_array: ",coarsen_workload_array)
    coarsen_samples = coarse_height * coarse_width
    coarsen_samples_list = list(range(coarsen_samples))
    if extra != -1:
        random.seed(extra)
        random.shuffle(coarsen_samples_list)
    random_choice = 0
    print("coarsen_workload_array: \n",coarsen_workload_array)
    # print("coarsen_samples_list: ",coarsen_samples_list)
    for coarsen_sample in coarsen_samples_list:
        min_processor = -1
        # the coordinates of the samples in the coarsen blocks 
        coarse_x, coarse_y = itowh(coarsen_sample, coarse_width, coarse_height)
        # first level neighbors
        # left
        left = whtoi(coarse_x - 1, coarse_y, coarse_width, coarse_height)
        # right
        right = whtoi(coarse_x + 1, coarse_y, coarse_width, coarse_height)
        # up
        up = whtoi(coarse_x, coarse_y - 1, coarse_width, coarse_height)
        # down
        down = whtoi(coarse_x, coarse_y + 1, coarse_width, coarse_height)
        # the list of neighbors
        coarse_neighbors = [left, right, up, down]
        samples = coarse_dictionary[coarsen_sample]
        print("samples: ",samples)
        neighbors = []
        # print("coarse_neighbors: ",coarse_neighbors)
        for coarsen_neighbor in coarse_neighbors:
                neighbors.append(coarse_dictionary[coarsen_neighbor])
        flat_list = []
        for sublist in neighbors:
            for item in sublist:
                flat_list.append(item)
        neighbors = flat_list
        print("neighbors: ",neighbors)
        arr = np.zeros(processors)
        for neighbor in neighbors:
            if assignment[neighbor] != -1:
                arr[assignment[neighbor]] += 1
        dictionary = dict(zip(list(range(processors)), arr))
        dictionary = {k: v for k, v in dictionary.items() if v != 0}
        dictionary = dict(sorted(dictionary.items(), key = lambda item: item[1], reverse = True))
        ranked_array = list(dictionary.keys())
        print("ranked_array: ",ranked_array)
        for ranked_processor in ranked_array:
            threshold = copy.deepcopy(processor_workload_array)[ranked_processor].reshape(-1,1)
            for sample in samples:
                threshold += workload_array[sample].reshape(-1, 1)
            # print("threshold: ",threshold)
            is_overloaded = False
            for index in range(len(threshold)):
                if(threshold[index]>computation_overload_threshold[index]):
                    is_overloaded = True
                    print("overloaded processor ", ranked_processor)
                    break
            if is_overloaded:
                continue
            min_processor = ranked_processor
            break
        # print("assigned samples: ",samples)
        # if min_processor is found
        if min_processor != -1:
            for sample in samples:
                assignment[sample] = min_processor
            print("coarse assignment\n",assignment)
            print("min_processor: ",min_processor)
        else: 
            if (single_entry):
                # assign individually
                for sample in samples:
                    min_cost = float("inf")
                    for processor in range(processors):
                        # uncoarsen the block
                        tmp_processor_workload_array = copy.deepcopy(processor_workload_array)
                        tmp_processor_workload_array[processor] += workload_array[sample]
                        tmp_cost = np.sum(np.max(tmp_processor_workload_array, axis = 0))
                        if min_cost > tmp_cost:
                            min_processor = processor
                            min_cost = tmp_cost
                        # coarsen the block
                    random_choice += 1
                    assignment[sample] = min_processor
                    print("greedy fine tune assignment\n",assignment)
                    print("min_processor: ",min_processor)
            else:
                # assign the entired block
                min_cost = float("inf")
                for processor in range(processors):
                    # tmp_processor_workload_array = processor_workload_array
                    tmp_processor_workload_array = copy.deepcopy(processor_workload_array)
                    for sample in samples:
                        tmp_processor_workload_array[processor] += workload_array[sample]
                    tmp_cost = np.sum(np.max(tmp_processor_workload_array, axis = 0))
                    # print("tmp_cost of processor ", processor, " is ",tmp_cost)
                    if min_cost > tmp_cost:
                        print("min_processor_updated to ",processor)
                        min_processor = processor
                        min_cost = tmp_cost
                    # coarsen the block
                random_choice += 1
                for sample in samples:
                    assignment[sample] = min_processor
                print("greedy block assignment\n", assignment)
                print("min_processor: ",min_processor)
        # print("processor_workload_array: \n",processor_workload_array)
        for sample in samples:
            np.add(processor_workload_array[min_processor], workload_array[sample], out=processor_workload_array[min_processor])
        # print("processor_workload_array: \n",processor_workload_array)
        print("\n")
    print("computation_overload_threshold\n", computation_overload_threshold)
    total = width * height
    print("greedy choice percentage: ",random_choice/total)
    return assignment

def itowh(i: int, width: int, height: int) -> Tuple[int, int]:
    if i < 0 or i >= width * height:
        raise ValueError('i out of bounds')
    return i % width, int(i / width)

# Given the coordinate in the un-coarsened block, return the nearby neighbor of a coarsened size block
def get_block_index(coarse, x, y, width, height):
    blocks = []
    const = math.floor(coarse/2)
    # print("const is ",const)
    starting_x = x + const
    starting_y = y + const
    for i in range(coarse):
        for j in range(coarse):
            blocks.append(whtoi(x+i, y-j, width, height))
    return blocks

# Given a index in the un-coarsened block, return the index in the coarsened block
# def get_coarsen_index():


# Given a index in the coarsened block, return the list of indices in the un-coarsened block (inverse function)
def get_uncoarsen_indices(coarsen_index, coarsen_width, coarsen_height, coarse):
    coarsen_x, coarsen_y = itowh(coarsen_index, coarsen_width, coarsen_height)
    coarsen_x *= coarse
    coarsen_y *= coarse
    print("coarsen_x: ",coarsen_x)
    print("coarsen_y: ",coarsen_y)
    retVal = []
    return whtoi

def reshape_to_3d(matrix_2d, width, height, interval):
    print("width: ",width)
    print("height: ",height)
    matrix_3d = [[[0]*interval for _ in range(height)] for _ in range(width)]
    for i in range(width):
        for j in range(height):
            for k in range(interval):
                print("i=",i,"j=",j,"k=",k)
                matrix_3d[i][j][k] = matrix_2d[j * width + i][k]
    return matrix_3d

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

def reshape_to_2d(matrix_3d):
    height, width, interval = len(matrix_3d), len(matrix_3d[0]), len(matrix_3d[0][0])
    matrix_2d = [matrix_3d[i//width][i%width] for i in range(height*width)]
    return matrix_2d

def coarsen(matrix_2d, width, height, interval,coarse_constant):
    # Calculate the coarsened dimensions
    coarse_width = math.ceil(width/coarse_constant)
    coarse_height = math.ceil(height/coarse_constant)
    print(coarse_width,coarse_height)
    coarse_matrix = [[[0]*interval for _ in range(coarse_height)] for _ in range(coarse_width)]
    new_dictionary = {}
    for i in range(coarse_width):
        for j in range(coarse_height):
            tuple_key = (i,j)
            index = whtoi(i, j, coarse_width, coarse_height)
            new_dictionary[index] = []

    for y in range(height):
        j = math.floor(y/coarse_constant)
        for x in range(width):
            i = math.floor(x/coarse_constant)
            tuple_value = whtoi(x, y, width, height)
            new_dictionary[whtoi(i, j, coarse_width, coarse_height)].append(tuple_value)
            for k in range(interval):
                coarse_matrix[i][j][k] += matrix_2d[whtoi(x, y, width, height)][k]
    # Convert coarsened 3D matrix back to 2D
    coarse_matrix_2d = reshape_to_2d(coarse_matrix)
    return coarse_matrix_2d, coarse_width, coarse_height, new_dictionary

def greedy_prioritize_communication(matrix: list, width: int, height: int, intervals: int, processors: int, extra: int, send_cost: float, recv_cost: float, constant: float) -> List[int]:
    # print("Here is matrix: \n", matrix)
    samples = height * width
    assignment = [-1] * samples
    processor_workload_array = np.zeros(processors * intervals).reshape(processors, intervals)
    workload_array = np.array(matrix)
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
        # if min_processor is found    
        if min_processor != -1:
            assignment[sample] = min_processor
            print("coarse assignment")
        else: 
            min_cost = float("inf")
            for processor in range(processors):
                # uncoarsen the block
                tmp_processor_workload_array = copy.deepcopy(processor_workload_array)
                tmp_processor_workload_array[processor] += workload_array[sample]
                tmp_cost = np.sum(np.max(tmp_processor_workload_array, axis = 0))
                if min_cost > tmp_cost:
                    min_processor = processor
                    min_cost = tmp_cost
                # coarsen the block
            random_choice += 1
            assignment[sample] = min_processor
            print("greedy assignment")
            # print("min_processor: ",min_processor)
            # print("assignment\n",assignment)
        # print("processor_workload_array\n",processor_workload_array)
        processor_workload_array[min_processor] += workload_array[sample]
    # print("processor_workload_array",processor_workload_array)
    print("computation_overload_threshold\n", computation_overload_threshold)
    print("greedy choice percentage: ",random_choice/samples)
    # print("Number of greedy assignment: ",random_choice)
    return assignment

# tuple_key = whtoi(i, j, width, height)
# tuple_value = whtoi(x, y, width, height)
# coarse_dictionary[tuple_key] = tuple_value
def main():
    width = 7
    height = 7
    intervals = 4
    processors = 3
    extra = -1
    send_cost = 2
    recv_cost = 2
    constant = 1.3
    coarse = 3
    coarsen_width = 3
    coarsen_height = 3
    file_path = 'random_workload.csv'
    # with open(file_path, 'r') as file:
    #     csv_reader = csv.reader(file)
    #     width = sum(1 for row in csv_reader)
    samples = width * height
    # read into the workload matrix and update interval
    workload_matrix = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=' ')
        for row in csv_reader:
            if len(row) > 0:
                row = [int(element) for element in row]
                intervals = len(row)
                workload_matrix.append(row)
    # print("length of workload: ",len(workload_matrix))
    # print("original workload: \n",workload_matrix)
    # final_matrix, constant, final_const, coarse_dictionary = coarsen(workload_matrix, width, height, intervals, 2)
    # print("processed workload: \n", coarsen(workload_matrix, width, height, intervals, 2))
    # print("coarse dictionary: \n",coarse_dictionary)s
    extra = -1
    print("assignment: ",greedy_prioritize_communication_fine_tune(workload_matrix, width, height, intervals, processors, extra, send_cost, recv_cost, constant, coarse, False))
    # print("assignment: ",greedy_prioritize_communication(workload_matrix, width, height, intervals, processors, extra, send_cost, recv_cost, constant, coarse))
    # print(get_uncoarsen_indices(2, coarsen_width, coarsen_height, 2))
    # print(get_block_index(2, 2, 2, width, height))
    # blocks = np.array([1, 2, 3, 4, 5])
    # sample_lists = np.array([2, 4, 6, 8, 5])
    # mask = np.isin(blocks, sample_lists)
    # blocks_filtered = blocks[mask]
    # print(blocks_filtered)
    # assignment = greedy_prioritize_communication_single_entry(workload_matrix, width, height, intervals, processors, extra, send_cost, receive_cost, constant, coarse)
    # print("processed workload: \n", coarsen(workload_matrix, width, height, intervals, 2))
    # # print("processed workload: \n",reshape_to_3d(workload_matrix, width, height, intervals))
    # print("coarsen workload: \n",coarsen(workload_matrix, width, height, intervals ,2))
    # print("processed workload: \n",reshape_to_2d(coarsen(workload_matrix,width,height,intervals,2)))
if __name__ == '__main__':
    main()