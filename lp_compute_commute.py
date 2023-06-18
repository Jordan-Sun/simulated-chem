import numpy as np
from scipy.optimize import minimize
from complex_assignment import get_neighbor
from complex_assignment import whtoi
from complex_assignment import itowh
from typing import List, Tuple
# This lp solver will consider both the communication and computation time and try to minimize the total time
# variables:
# samples: s
# intervals: t
# processors: p
# assignment_matrix (s x p)
# workload_matrix (t x s=)
# compute_cost_matrix (t x 1)
# communicate_cost_matrix (t x 1)

def lp_compute_commute(send_cost: int, receive_cost: int, width: int, height: int, workload_matrix: list, samples: int, intervals: int, processors: int) -> List[List[float]]:
    workload_matrix = np.array(workload_matrix)
    assignment_matrix = np.eye(processors)[np.random.choice(processors, samples)]
    assignment_matrix = assignment_matrix.reshape((-1, processors))

    workload_matrix = np.array(workload_matrix)[:samples, :intervals]
    # print("samples: ",samples)
    # print("intervals: ",intervals)
    # print("processors: ",processors)
    # print("workload_matrix shape: ",workload_matrix.shape)
    # print("workload_matrix: ",workload_matrix)
    #conversion into numpy matrix

    def objective(assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors):
        # computation only
        assignment_matrix = assignment_matrix.reshape(-1,processors)
        compute_time_matrix = np.array(workload_matrix).T @ assignment_matrix
        compute_cost = np.sum(np.max(compute_time_matrix, axis=1).reshape(-1, 1))
        total_cost = compute_cost
        communicate_time = np.zeros(processors)
        communicate_cost = 0
        # communication only
        for p in range(processors):
            tasks_array = np.nonzero(assignment_matrix[:, p])[0]
            # print("processor ",p," task_array: ", tasks_array)
            for task in range(samples):
                neighbor_list = get_neighbor(task, width, height)
                # print("processor ", p," neighbor_list: ",neighbor_list)
                for neighbor_index in neighbor_list:
                        communicate_time[p] += (send_cost + receive_cost) * assignment_matrix[task][p]
        communicate_cost = np.max(communicate_time)
        # print("communication_time: ",communicate_time)
        total_cost += communicate_cost * intervals
        print("total cost: ",total_cost)
        return total_cost

        # communicate_cost = np.max(communicate_time)
        # retVal = np.sum(compute_cost) + communicate_cost * intervals
        # print("retVal = ",retVal)
        retVal = np.sum(compute_cost)
        print (retVal)
        return retVal

    # A task can only be partially assigned to one processor
    def linear_constraint(assignment_matrix):
        assignment_matrix = assignment_matrix.reshape(-1, processors)  # Reshape the input matrix
        row_sums = np.sum(assignment_matrix, axis=1)
        return row_sums - 1
    # partial assignment of a task to a processor must be non-negative
    def nonnegativity_constraint(x):
        matrix = x.reshape((samples, processors))
        return matrix.flatten()
    
    constraint = ({'type': 'eq', 'fun': linear_constraint}, {'type': 'ineq', 'fun': nonnegativity_constraint})
    options = {'maxiter': 1000000}
    tolerance = 100
    result = minimize(objective, matrix, options = options, args = (send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors), constraints = constraint, tol = tolerance)
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
