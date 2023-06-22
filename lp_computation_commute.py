import numpy as np
from scipy.optimize import minimize
from complex_assignment import get_neighbor
from complex_assignment import whtoi
from complex_assignment import itowh
import random
from typing import List, Tuple
# This lp solver will consider both the communication and computation time and try to minimize the total time
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

def lp_compute_commute(matrix: np.ndarray, send_cost: int, receive_cost: int, width: int, height: int, workload_matrix: list, samples: int, intervals: int, processors: int) -> List[List[float]]:

    workload_matrix = np.array(workload_matrix).reshape(-1,intervals)
    # print("samples: ",samples)
    # print("intervals: ",intervals)
    # print("processors: ",processors)
    # print("workload_matrix shape: ",workload_matrix.shape)
    # print("workload_matrix: ",workload_matrix)
    #conversion into numpy matrix

    def objective(assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors):
        # computation only
        assignment_matrix = assignment_matrix.reshape(-1,processors)
        compute_cost = 0
        total_cost = 0
        compute_time_matrix = np.array(workload_matrix).T @ assignment_matrix
        compute_cost = np.sum(np.max(compute_time_matrix, axis=1).reshape(-1, 1))
        total_cost += compute_cost
        communicate_time = np.zeros(processors)
        communicate_cost = 0
        # communication only
        for p in range(processors):
            # Method 1:
            tasks_array = []
            for i in range(samples):
                if (np.argmax(assignment_matrix[i]) == p):
                    tasks_array.append(i)
            # Method 2:
            # tasks_array = np.nonzero(assignment_matrix[:, p])[0]
            for task in tasks_array:
                # print("task: ",task)
                task_neighbor_array = get_neighbor(task, width, height)
                for neighbor in task_neighbor_array:
                    # Compute communication cost of a neighbor
                    # Method 1: 
                    # communicate_time[p] += (send_cost + receive_cost) * (np.argmax(assignment_matrix[neighbor]) != np.argmax(assignment_matrix[task]))
                    # Method 2:
                    communicate_time[p] += (send_cost + receive_cost) * (1-assignment_matrix[neighbor][p])
        communicate_cost = np.max(communicate_time)
        # print("communicate time of processor: ",communicate_time)
        # print("communication_time: ",communicate_cost)
        total_cost += communicate_cost * intervals
        # print("total cost: ",total_cost)
        return total_cost

    # A task can only be partially assigned to one processor
    def linear_constraint(assignment_matrix):
        assignment_matrix = assignment_matrix.reshape(-1, processors)  # Reshape the input matrix
        row_sums = np.sum(assignment_matrix, axis=1)
        return row_sums - 1

    # partial assignment of a task to a processor must be non-negative
    def nonnegativity_constraint(x):
        matrix = x.reshape((samples, processors))
        return matrix.flatten()
        
    # intialize assignment_matrix
    # matrix = np.zeros((samples,processors))
    # for i in range(samples):
    #     matrix[i][np.random.randint(0, processors)] =  1
    # print("matrix: ",matrix)
    constraint = ({'type': 'eq', 'fun': linear_constraint}, {'type': 'ineq', 'fun': nonnegativity_constraint})
    tolerance = 0.01
    max_iterations = 100
    options = {'maxiter': max_iterations}
    result = minimize(objective, matrix,options = options, args = (send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors), constraints = constraint, tol = tolerance)
    # print("result: \n",result)
    return (result.x.reshape(-1,processors)).tolist()

# convert lp assignment to assignment by max
def lp_compute_commute_max(solution: list, samples: int, processors: int) -> list:
    # construct assignment
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