import numpy as np
from scipy.optimize import minimize
from complex_assignment import get_neighbor
from complex_assignment import whtoi
from complex_assignment import itowh
import random
from scipy.optimize import linprog
from typing import List, Tuple
# variables:
# samples: s
# intervals: t
# processors: p
# assignment_matrix (samples x processors)
# workload_matrix: samples x intervals
# compute_cost_matrix (intervals x 1)
# communicate_cost_matrix (intervals x 1)

def get_dimensions(lst):
    dimensions = []
    while isinstance(lst, list):
        dimensions.append(len(lst))
        lst = lst[0] if lst else None
    return dimensions

def get_tasks(tasks_array, assignment_matrix, processors, samples):
    for i in range(samples):
        if (np.argmax(assignment_matrix[i]) == p):
            tasks_array.append(i)

# def get_communication_cost_heuristic_one():
    
# def get_communication_cost_heuristic_two():

# def get_communication_cost_heuristic_three():

def get_computation_cost(assignment_matrix, workload_matrix):
    compute_time_matrix = np.array(workload_matrix).T @ assignment_matrix
    compute_cost = np.sum(np.max(compute_time_matrix, axis=1).reshape(-1, 1))
    return compute_cost

def get_communication_cost(assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors):
    communicate_time = np.zeros(processors)
    communicate_cost = 0
    # communication only
    for p in range(processors):
        # Method 1:
        tasks_array = []
        # tasks_array = get_tasks(tasks_array, assignment_matrix, processors, samples)
        for i in range(samples):
            if (np.argmax(assignment_matrix[i]) == p):
                tasks_array.append(i)
        # Method 2:
        # tasks_array = np.nonzero(assignment_matrix[:, p])[0]
        for task in tasks_array:
            # print("task: ",task)
            task_neighbor_array = get_neighbor(task, width, height)
            for neighbor in task_neighbor_array:
                # Method 1: 
                communicate_time[p] += (send_cost + receive_cost) * (np.argmax(assignment_matrix[neighbor]) != np.argmax(assignment_matrix[task]))
                # Method 2:
                # communicate_time[p] += (send_cost + receive_cost) * (1-assignment_matrix[neighbor][p])
    # communicate_cost = np.max(communicate_time)
    return communicate_time

def lp_compute_commute(assignment_matrix: np.ndarray, send_cost: int, receive_cost: int, width: int, height: int, workload_matrix: list, samples: int, intervals: int, processors: int) -> List[List[float]]:
    workload_matrix = np.array(workload_matrix).reshape(-1,intervals)
    # print("samples: ",samples)
    # print("intervals: ",intervals)
    # print("processors: ",processors)
    # print("workload_matrix shape: ",workload_matrix.shape)
    # print("workload_matrix: ",workload_matrix)
    #conversion into numpy matrix
    print("hi3")
    def objective(assignment_matrix, communication_cost_array, communicate_cost):
        total_cost = np.sum(communication_cost_array) + communicate_cost
        print("hi2")
        # assignment_matrix = assignment_matrix.reshape(-1,processors)
        # compute_cost = get_computation_cost(assignment_matrix, workload_matrix)
        # communicate_cost = get_communication_cost(assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors)
        # total_cost += compute_cost
        # total_cost += communicate_cost * intervals

        # print("communication_time: ",communicate_cost)
        # print("total cost: ",total_cost)
        return total_cost 

    # A task can only be partially assigned to one processor
    def linear_constraint(assignment_matrix):
        assignment_matrix = assignment_matrix.reshape(-1, processors)  # Reshape the input matrix
        row_sums = np.sum(assignment_matrix, axis=1)
        return row_sums - 1

    # partial assignment of a task to a processor must be non-negative
    def nonnegativity_constraint(assignment_matrix):
        matrix = assignment_matrix.reshape((samples, processors))
        return matrix.flatten()

    def computation_cost_constraint(computation_cost_array, workload_matrix, assignment_matrix):
        constant_array = np.array(workload_matrix).T @ assignment_matrix
        print(constant_array.shape)
        differences = computation_cost_array.reshape((-1,processors)) - constant_array
        return differences.flatten()
    
    def communication_cost_constraint(communication_cost, assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors):
        constant_array = get_communication_cost(assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors)
        return communication_cost - constant_array.flatten()

    # intialize assignment_matrix
    # matrix = np.zeros((samples,processors))
    # for i in range(samples):
    #     matrix[i][np.random.randint(0, processors)] =  1
    # print("matrix: ",matrix)
    compute_cost_array = np.ones((1,intervals))
    communication_cost = 9
    constraint = ({'type': 'eq', 'fun': linear_constraint}, {'type': 'ineq', 'fun': nonnegativity_constraint}, {'type': 'ineq', 'fun': lambda x: computation_cost_constraint(x, workload_matrix, assignment_matrix)}, {'type': 'ineq', 'fun': lambda x: communication_cost_constraint(x, communication_time)})
    tolerance = 0.00001
    max_iterations = 100
    print("hi1")
    options = {'maxiter': max_iterations}
    communicate_cost = get_communication_cost(assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors)
    result = minimize(objective, assignment_matrix, args = (intervals, compute_cost_array, communication_cost, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors), options = options, constraints = constraint, tol = tolerance)
    print("hi")
    # print("result: \n",result)
    return (result.x.reshape(-1,processors)).tolist()

# convert lp assignment to assignment by max
def lp_compute_commute_max(solution: list, samples: int, processors: int) -> list:
    result = []
    for i in range(samples):
        assignment = 0
        value = 0
        for k in range(processors):
            if solution[i][k] > value:
                assignment = k
                value = solution[i][k]
        result.append(assignment)
    return result

# convert lp assignment to assignment by randomization based on solution
def lp_compute_commute_random(solution: list, samples: int, processors: int) -> list:
    result = []
    for i in range(samples):
        weights = [solution[i][k] for k in range(processors)]
        result.append(random.choices(range(processors), weights = weights)[0])
    return result