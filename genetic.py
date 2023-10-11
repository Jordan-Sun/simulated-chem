import simulation
import numpy as np
import pandas as pd
import time
import os
import random
import csv
import copy
import complex_assignment
import simple_assignment
import sys
from typing import List

def cord_to_index(width: int, _: int, x: int, y: int) -> int:
    return x * width + y

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

def simulate_genetic_alg(matrix: list, assignment: list, width: int, height: int, intervals: int, processors: int, sendCost: int, recvCost: int):
    # Computation costs
    samples = width * height
    helper_assignment_matrix = np.zeros((samples, processors))
    for sample in range(samples):
        helper_assignment_matrix[sample][assignment[sample]] = 1
    compute_time_matrix = helper_assignment_matrix.T @ np.array(matrix)
    max_compute_time = [max(column) for column in zip(*compute_time_matrix)]
    totalComputation = sum(max_compute_time)
    # Broadcast costs
    totalBroadcast = 0
    # Process each interval
    for interval in range(intervals):
        # Find the processor with the maximum work.
        intervalBroadcasts = [0] * processors
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
                intervalBroadcasts[assignment[sample]] += sendCost * min(1, different) + recvCost * different
        # Add the maximum work to the total.
        totalBroadcast += max(intervalBroadcasts)
    return totalBroadcast + totalComputation

def get_theoretical_lower_bound(width, height, input_matrix, denominator):
    value = 0
    column_max_values = [max(col) for col in zip(*input_matrix)]
    column_average_values = np.mean(input_matrix, axis=0)
    column_average_values = [x / denominator for x in column_average_values]
    for t in range(width):
        value += max(column_max_values[t], column_average_values[t])
    return value

def calculate_fitness(workload_matrix, assignment_matrix, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound):
    totalComputationBroadcast = simulate_genetic_alg(workload_matrix, assignment_matrix, width, height, intervals, processors, send_cost, receive_cost)
    return -totalComputationBroadcast

def select_chromosomes(population, workload_matrix, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound):
    denominator = sum(chromosome[1] for chromosome in population)
    fitness_values = [float(chromosome[1]) / denominator for chromosome in population]
    # print("fitness_values: ",fitness_values)
    parent1 = random.choices(population, weights=fitness_values, k=1)[0]
    parent2 = random.choices(population, weights=fitness_values, k=1)[0]
    # print("Selected two chromosomes for crossover")
    return parent1, parent2

def crossover(parent1, parent2, workload_matrix, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound, mutation_rate):
    child1_matrix = copy.deepcopy(parent1[0])
    child2_matrix = copy.deepcopy(parent2[0])
    random_list = random.sample(range(0, len(child1_matrix)), k = int(len(child1_matrix) * mutation_rate))
    for i in range(len(child1_matrix)):
        if i in random_list:
            tmp_value = child1_matrix[i]
            child1_matrix[i] = child2_matrix[i]
            child2_matrix[i] = tmp_value
    child1 = [child1_matrix, calculate_fitness(workload_matrix, child1_matrix, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound)]
    child2 = [child2_matrix, calculate_fitness(workload_matrix, child2_matrix, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound)]
    # print("Performed crossover between two chromosomes")
    return child1, child2

def mutate(chromosome, processors, mutation_rate):
    random_list = random.sample(range(0, len(chromosome[0])), k = int(len(chromosome[0]) * mutation_rate))
    for mutation_point in range(len(random_list)):
        chromosome[0][mutation_point] = random.randint(0, processors-1)
    print("Performed mutation on a chromosome")
    return chromosome

# function to get the best chromosome from the population
def get_best(population):
    max_index = max(range(len(population)), key=lambda i: population[i][1])
    return population[max_index]

def generate_population(size, workload_matrix, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound):
    population = []
    for index in range(size):
        print("index = ", index)
        # randomly add greedily generated chromosome or randomly generated chromosome
        # if random.uniform(0, 1) < 0:
        #     assignment = complex_assignment.greedy(workload_matrix, width, height, intervals, processors, index)
        #     score = calculate_fitness(workload_matrix, assignment, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound)
        #     chromosome = [assignment, score]
        #     population.append(chromosome)
        # else:
        #     pass
        assignment = [random.randint(0, processors-1) for _ in range(width * height)]
        score = calculate_fitness(workload_matrix, assignment, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound)
        chromosome = [assignment, score]
        population.append(chromosome)
    print("Generated a random population of size", size)
    return population

def genetic_algorithm(workload_matrix: list, width: int, height: int, intervals: int, processors: int, extra: int, send_cost: float, receive_cost: float) -> List[int]:
    # parameters
    print("width = ",width)
    print("height = ",height)
    samples  = width * height
    r = 2
    trial = 1
    generation = 20000
    population_size = 700
    mutation_probability = 0.05
    mutation_rate = 0.02
    start_time = time.time()
    ####### read workload
    # workdir = os.path.join('output', 'x_{}'.format(width), 'y_{}'.format(height), 't_{}'.format(intervals), 'r_{}'.format(r), 'trial_{}'.format(trial))
    # input_name = os.path.join(workdir, 'workload.csv')
    # if not os.path.exists(input_name):
    #     print('Workload file does not exist')
    #     return -4
    # # read the workload from the file
    # workload_matrix = pd.read_csv(input_name, header='infer', index_col=0).values.tolist()
    # samples = width * height
    # if len(workload_matrix) != samples:
    #     print('Workload file has invalid number of samples')
    #     return -5
    #######
    theoretical_lower_bound = get_theoretical_lower_bound(intervals, samples, workload_matrix, processors)
    assignment_matrix = None
    # perform genetic algorithms
    population = generate_population(population_size, workload_matrix, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound)
    gen = 0
    current_best = get_best(population)
    initial_total_computation, initial_total_broadcast, initial_total_unicast, initial_total_computation_broadcast, initial_total_computation_unicast = simulation.simulate(workload_matrix, current_best[0], width, height, intervals, processors, send_cost, receive_cost)
    best_efficiency = float('-inf')
    best_offspring = []
    # natural selection
    while gen < generation and best_efficiency < 1/3:
        print("Generation ", gen+1)
        gen += 1
        # for i in range(len(population)):
        #     print(f"population [{i}]: ",population[i])
        # select two chromosomes for crossover
        # print("width: ",width)
        # print("height: ",height)
        # print("processors: ",processors)
        # print("send cost: ",send_cost)
        # print("receive cost: ",receive_cost)
        # print("intervals: ",intervals)
        # print("assignment matrix: ", assignment_matrix)
        # print("parent1: ",parent1)
        # print("parent2: ",parent2)
        # perform crossover to generate two new chromosomes
        # random selection
        parent1, parent2 = select_chromosomes(population, workload_matrix, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound)
        # elitism selection on one parent
        parent2 = get_best(population)
        # elitism selection on both parents
        # population_copy = [x for x in population]
        # population_copy.remove(parent2)
        # parent1 = get_best(population_copy)
        # perform crossover
        child1, child2 = crossover(parent1, parent2, workload_matrix, width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound, mutation_rate)
        
        # perform mutation on the two new chromosomes
        if random.uniform(0, 1) < mutation_probability:
            child1 = mutate(child1, processors, mutation_rate)
        if random.uniform(0, 1) < mutation_probability:
            child2 = mutate(child2, processors, mutation_rate)
        # print("child 1: ",child1)
        # print("child 2: ",child2)
        # replace the old population with the new population
        element1 = [child1[0], calculate_fitness(workload_matrix, child1[0], width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound)]
        element2 = [child2[0], calculate_fitness(workload_matrix, child2[0], width, height, intervals, processors, send_cost, receive_cost, theoretical_lower_bound)]
        population.append(element1)
        population.append(element2)
        # find the best among all population
        best_offspring = get_best(population)
        best_efficiency = best_offspring[1]
        # print("best_efficiency: ",best_efficiency)
        # print("best_offspring: ",best_offspring)
        totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast = simulation.simulate(workload_matrix, best_offspring[0], width, height, intervals, processors, send_cost, receive_cost)
        print(totalComputationBroadcast)
    # evaluate the best chromosome from the population
    print("Initial total Computation: ",initial_total_computation)
    print("Initial total Broadcast: ",initial_total_broadcast)
    print("Initial total Unicast: ",initial_total_unicast)
    print("Initial total Computation Broadcast: ",initial_total_computation_broadcast)
    print("Initial total Computation Unicast: ",initial_total_computation_unicast)
    print("theoretical_lower_bound: ",theoretical_lower_bound)
    print("efficiency ratio: ", "{:.3f}".format(initial_total_computation/theoretical_lower_bound))
    best_offspring = get_best(population)
    totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast = simulation.simulate(workload_matrix, best_offspring[0], width, height, intervals, processors, send_cost, receive_cost)
    print("Final total Computation: ",totalComputation)
    print("Final total Broadcast: ",totalBroadcast)
    print("Final total Unicast: ",totalUnicast)
    print("Final total Computation Broadcast: ",totalComputationBroadcast)
    print("Final total Computation Unicast: ",totalComputationUnicast)
    print("theoretical_lower_bound: ",theoretical_lower_bound)
    print("efficiency ratio: ", "{:.3f}".format(totalComputation/theoretical_lower_bound))
    end_time = time.time()
    print("Elapsed time:", end_time - start_time, "seconds")
    return best_offspring

if __name__ == '__main__':
    main()