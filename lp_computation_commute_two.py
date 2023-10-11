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

def get_computation_cost(assignment_matrix, workload_matrix):
    compute_time_matrix = np.array(workload_matrix).T @ assignment_matrix
    compute_cost = np.sum(np.max(compute_time_matrix, axis=1).reshape(-1, 1))
    return compute_cost

def get_task_heuristic_one(samples, assignment_matrix, p):
    tasks_array = []
    for i in range(samples):
        if (np.argmax(assignment_matrix[i]) == p):
            tasks_array.append(i)
    return tasks_array

def get_task_heuristic_two(assignment_matrix, p):
    tasks_array = np.nonzero(assignment_matrix[:, p])[0]
    return tasks_array

def communication_cost_heuristic_one(p, send_cost, receive_cost, assignment_matrix, neighbor, task):
    return (send_cost + receive_cost) * (np.argmax(assignment_matrix[neighbor]) != np.argmax(assignment_matrix[task]))

def communication_cost_heuristic_two(p, send_cost, receive_cost, assignment_matrix, neighbor):
    return (send_cost + receive_cost) * (not (assignment_matrix[neighbor][p]))

def communication_cost_heuristic_three(p, send_cost, receive_cost, assignment_matrix, neighbor):
    return (send_cost + receive_cost) * (1-assignment_matrix[neighbor][p])

def get_communication_cost(assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors):
    communicate_time = np.zeros(processors)
    communicate_cost = 0
    # communication only
    for p in range(processors):
        # Method 1:
        # tasks_array = get_task_heuristic_one(samples, assignment_matrix, p)
        # Method 2:
        tasks_array = get_task_heuristic_two(assignment_matrix, p)
        print(tasks_array, " are tasks of processor ",p)
        # print("processor ",p, " has ", len(tasks_array), " tasks out of ", samples," tasks")
        for task in tasks_array:
            # print("task: ",task)
            task_neighbor_array = get_neighbor(task, width, height)
            for neighbor in task_neighbor_array:
                # Method 1: 
                # communicate_time[p] += communication_cost_heuristic_one(p, send_cost, receive_cost, assignment_matrix, neighbor, task)
                # Method 2:
                communicate_time[p] += communication_cost_heuristic_two(p, send_cost, receive_cost, assignment_matrix, neighbor)
                # Method 3:
                # communicate_time[p] += communication_cost_heuristic_three(p, send_cost, receive_cost, assignment_matrix, neighbor)
                pass
    # communicate_cost = np.max(communicate_time)
    return communicate_time

def lp_compute_commute(assignment_matrix: np.ndarray, send_cost: int, receive_cost: int, width: int, height: int, workload_matrix: list, samples: int, intervals: int, processors: int) -> List[List[float]]:
    
    workload_matrix = np.array(workload_matrix).reshape(-1,intervals)
    # print("workload_matrix shape: ",workload_matrix.shape)
    # print("assignment_matrix shape: ",assignment_matrix.shape)

    def objective(variables):
        total = 0
        communication_cost = variables[-1]
        compute_cost_array = variables[samples * processors : samples * processors + intervals]
        total += communication_cost
        computation_cost = np.sum(compute_cost_array)
        total += computation_cost
        print("communication cost: ",communication_cost, " computation cost: ", computation_cost)
        return total

    def linear_constraint(variables):
        extracted_assignment_matrix = variables[0 : samples * processors].reshape((samples, processors))
        assignment_matrix = extracted_assignment_matrix.reshape(-1, processors)  # Reshape the input matrix
        row_sums = np.sum(assignment_matrix, axis=1)
        return row_sums - 1

    def nonnegativity_constraint(variables):
        assignment_matrix = variables[0 : samples * processors].reshape((samples, processors))
        return assignment_matrix.flatten()

    def computation_cost_constraint(variables, workload_matrix):
        compute_cost_array = variables[samples * processors : samples * processors + intervals]
        assignment_matrix = variables[0 : samples * processors].reshape((samples, processors))
        # print("compute_cost_array in computation_cost_constraint: ",compute_cost_array)
        constant_matrix = (np.array(workload_matrix).T @ assignment_matrix).T
        # print("computation cost distribution matrix: ",constant_matrix)
        constraints = []
        for row in constant_matrix:
            constraints.append(compute_cost_array - row)
        return np.concatenate(constraints)

    def communication_cost_constraint(variables, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors):
        communication_cost = variables[-1]
        # print("communication cost in communication_cost_constraint: ",communication_cost)
        assignment_matrix = variables[0 : samples * processors].reshape((samples, processors))
        # print("assignment_matrix in communication_cost_constraint: ",assignment_matrix)
        constant_array = get_communication_cost(assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors)
        # print(communication_cost - constant_array.flatten())
        return communication_cost - constant_array.flatten()

    constraint_linear = {'type': 'eq', 'fun': linear_constraint, 'args': ()}
    constraint_nonnegativity = {'type': 'ineq', 'fun': nonnegativity_constraint, 'args': ()}
    constraint_communication_cost = {'type': 'ineq', 'fun': communication_cost_constraint, 'args': (send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors)}
    constraint_computation_cost = {'type': 'ineq', 'fun': computation_cost_constraint, 'args': (workload_matrix,)}
    constraints = [constraint_linear, constraint_nonnegativity, constraint_communication_cost, constraint_computation_cost]
    
    compute_cost_array = np.ones((1,intervals))
    communication_cost = 1000
    initial_guess = [assignment_matrix.flatten(), compute_cost_array.flatten(), np.array([communication_cost])] # assignment, computation_cost_matrix, communication_cost

    tolerance = 0.0001
    max_iterations = 10000
    options = {'maxiter': max_iterations}
    result = minimize(objective, np.concatenate(initial_guess), options = options, constraints = constraints, tol = tolerance)
    # print(result)
    # print(len(result.x))
    final_assignment = result.x[0 : samples * processors].reshape((samples, processors))
    return final_assignment.tolist()

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