'''
simple_assignment.py

This file contains the functions that will be used to assign the samples to the processors.
Each function takes in the width x and height y of the sample grid, and the number of processors.
Each cell in the list represents the processor that the sample is assigned to.
'''

import math
import random
from typing import List, Tuple

# to convert a 2d index to a 1d index
def cord_to_index(width: int, _: int, x: int, y: int) -> int:
    return x * width + y

# for proximity x assignment
def flipped_index_to_cord(_: int, height: int, i: int) -> Tuple[int, int]:
    return (i % height, i // height)

# for proximity matrix assignment
def index_to_cord(width: int, _: int, i: int) -> Tuple[int, int]:
    return (i // width, i % width)

# assigns the samples along the x axis, the samples have similar y coordinates
def proximity_y(width: int, height: int, p: int) -> List[int]:
    # the total number of samples is width times height
    n = width * height
    # the number of samples each processor will have
    samples_per_processor = n // p
    # the first remainder processors will have one more sample than the others
    remainder = n % p
    # the list of assignments
    assignments = [0] * n 
    # the last index that was assigned to a processor
    last_index = 0
    for i in range(p):
        # assign the samples to the processor
        for j in range(samples_per_processor):
            assignments[last_index + j] = i
        last_index += samples_per_processor
        # if the processor has one more sample than the others, assign it
        if i < remainder:
            assignments[last_index] = i
            last_index += 1
    return assignments

# assigns the samples along the y axis, the samples have similar x coordinates
def proximity_x(width: int, height: int, p: int) -> List[int]:
    # the total number of samples is width times height
    n = width * height
    # the number of samples each processor will have
    samples_per_processor = n // p
    # the first remainder processors will have one more sample than the others
    remainder = n % p
    # the list of assignments
    assignments = [0] * n 
    # the last index that was assigned to a processor
    last_index = 0
    for i in range(p):
        # assign the samples to the processor
        for j in range(samples_per_processor):
            x, y = flipped_index_to_cord(width, height, last_index + j)
            assignments[cord_to_index(width, height, x, y)] = i
        last_index += samples_per_processor
        # if the processor has one more sample than the others, assign it
        if i < remainder:
            x, y = flipped_index_to_cord(width, height, last_index)
            assignments[cord_to_index(width, height, x, y)] = i
            last_index += 1
    return assignments

# assigns the samples in submatrices, the samples have similar x and y coordinates
def proximity_matrix(width: int, height: int, p: int) -> List[int]:
    # the total number of samples is width times height
    n = width * height
    # the number of submatrices along the x axis
    submatrices_x = math.floor(math.sqrt(p))
    # the width of the submatrices
    submatrix_width = width // submatrices_x
    # the remainder of the division
    remainder_x = width % submatrices_x
    # the number of submatrices along the y axis
    submatrices_y = p // submatrices_x
    # the height of the submatrices
    submatrix_height = height // submatrices_y
    # the remainder of the division
    remainder_y = height % submatrices_y
    # the list of assignments
    assignments = [0] * n
    # x and y coordinates used
    last_x = 0
    last_y = 0
    # iterate through the submatrices
    for j in range(submatrices_y):
        for i in range(submatrices_x):
            # the processor that the submatrix is assigned to
            processor = j * submatrices_x + i
            # the width of the current submatrix
            current_submatrix_width = submatrix_width + (1 if i < remainder_x else 0)
            # the height of the current submatrix
            current_submatrix_height = submatrix_height + (1 if j < remainder_y else 0)
            # iterate through the samples in the submatrix
            for l in range(current_submatrix_height):
                for k in range(current_submatrix_width):
                    # assign the sample to the processor
                    index = cord_to_index(width, height, last_x + k, last_y + l)
                    # print('submatrice [{}, {}] ({}, {}) -> matrix ({}, {}): processor {} -> index {}'.format(i, j, k, l, last_x + k, last_y + l, processor, index))
                    assignments[index] = processor
            # update the last x coordinate
            last_x += current_submatrix_width
        # update the last x and y coordinates
        last_x = 0
        last_y += current_submatrix_height
    
    return assignments

# assigns the samples using a simple random assignment
def simple_random(width: int, height: int, p: int, ncell_max: List[int] = None) -> List[int]:
    # the total number of samples is width times height
    n = width * height
    # the list of assignments
    assignments = [0] * n

    # the pool of available processors
    pool = list(range(p))
    # validate ncell_max if provided
    if ncell_max is not None:
        if len(ncell_max) != p:
            raise RuntimeError('The length of ncell_max must be equal to the number of processors, but got {} for {} processors'.format(len(ncell_max), p))
    
    # assign each sample to a random processor
    for i in range(n):
        # randomly assign a processor from the pool of available processors
        assignments[i] = random.choice(pool)
        # remove the processor from the pool if it reaches the maximum number of samples
        if ncell_max is not None:
            ncell_max[assignments[i]] -= 1
            if ncell_max[assignments[i]] == 0:
                pool.remove(assignments[i])
    return assignments

# assigns the samples that tries to balance the number of samples per processor
def balanced_random(width: int, height: int, p: int) -> List[int]:
    # the total number of samples is width times height
    n = width * height
    # the number of rounds of random assignments is the ceiling of the number of samples divided by the number of processors
    rounds = math.ceil(n / p)
    # the list of unassigned samples
    unassigned = [i for i in range(n)]
    # the list of assignments
    assignments = [0] * n
    # randomly assign the samples to the processors in rounds
    for i in range(rounds):
        # randomly shuffle the unassigned samples
        random.shuffle(unassigned)
        # assign the samples to the processors
        for j in range(p):
            if len(unassigned) > 0:
                assignments[unassigned.pop()] = j
    return assignments
