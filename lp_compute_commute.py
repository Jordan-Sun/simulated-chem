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
# workload_matrix (t x s)
# compute_cost_matrix (t x 1)
# communicate_cost_matrix (t x 1)

def lp_compute_commute(send_cost: int, receive_cost: int, width: int, height: int, workload_matrix: list, samples: int, intervals: int, processors: int) -> List[List[float]]:

    # samples = 200
    # intervals = 20
    # processors = 16
    # workload_matrix = np.random.uniform(0,100,(samples, intervals))

    def objective(assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors):
        # print("workload_matrix shape: ", np.array(workload_matrix).shape)
        assignment_matrix = assignment_matrix.reshape(-1,processors)
        # print("assignment_matrix shape: ", assignment_matrix.shape)
        compute_time_matrix = np.array(workload_matrix).T @ assignment_matrix
        compute_cost = np.sum(np.max(compute_time_matrix, axis=1).reshape(-1, 1))
        print(compute_cost)
        return compute_cost

    matrix = np.zeros((samples,processors))
    for i in range(samples):
        matrix[i][np.random.randint(0, processors)] =  1
    print("matrix dimension: ",matrix.shape)
    print(matrix)

    # set constraint
    def linear_constraint(assignment_matrix):
        assignment_matrix = assignment_matrix.reshape(-1, processors)  # Reshape the input matrix
        row_sums = np.sum(assignment_matrix, axis=1)
        return row_sums - 1

    def nonnegativity_constraint(x):
        matrix = x.reshape((samples, processors))
        return matrix.flatten()
    
    constraint = ({'type': 'eq', 'fun': linear_constraint}, {'type': 'ineq', 'fun': nonnegativity_constraint})
    options = {'maxiter': 1000000}
    result = minimize(objective, matrix, options = options, args = (send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors), constraints = constraint)
    print("result: ",result)
    print("optimal assignment: ", result.x)

    # communicate_time = np.zeros(processors)
    # for p in range(processors):
    #     tasks_array = np.nonzero(reshaped_matrix[:, p])[0]
    #     for task in tasks_array:
    #         neighbor_list = get_neighbor(task, width, height)
    #         for neighbor_index in neighbor_list:
    #             if neighbor_index not in tasks_array:
    #                 communicate_time[p] += send_cost + receive_cost
    # communicate_cost = np.max(communicate_time)
