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

    print("workload_matrix shape: ", workload_matrix.shape)
    print("assignment_matrix shape:", assignment_matrix.shape)
    print("number of processor: ", processors)
    print("number of intervals: ", intervals)
    print("samples: ",samples)

    def compute_cost(matrix):
        reshaped_workload_matrix = np.reshape(workload_matrix, (samples, intervals))
        reshaped_matrix = np.reshape(matrix, (-1, processors))
        compute_time = reshaped_matrix.T @ reshaped_workload_matrix
        compute_cost = np.max(compute_time, axis=1).reshape(-1, 1)

    # def communicate_cost(matrix):

    def objective(matrix):
        reshaped_workload_matrix = np.reshape(workload_matrix, (samples, intervals))
        reshaped_matrix = np.reshape(matrix, (-1, processors))
        print("reshaped_workload_matrix shape: ",reshaped_workload_matrix.shape)
        print("reshaped_matrix shape: ",reshaped_matrix.shape)
        compute_time = reshaped_matrix.T @ reshaped_workload_matrix
        compute_cost = np.max(compute_time, axis=1).reshape(-1, 1)
        # communicate_time = np.zeros(processors)
        # for p in range(processors):
        #     # PSEUDO-CODE
        #     # task_array = get_task(p)
        #     # for task in task_array:
        #     #   neighbor_array = get_neighbor(task)
        #     #   for neighbor in neighbor_array:
        #     #     if neighbor not in task_array:
        #     #       communicate_time[p] += a+b (communicate)
        #     # get the indices of the tasks of processor p

        #     tasks_array = np.nonzero(reshaped_matrix[:, p])[0]
        #     for task in tasks_array:
        #         neighbor_list = get_neighbor(task, width, height)
        #         for neighbor_index in neighbor_list:
        #             if neighbor_index not in tasks_array:
        #                 communicate_time[p] += send_cost + receive_cost

        # communicate_cost = np.max(communicate_time)
        # retVal = np.sum(compute_cost) + communicate_cost * intervals
        # print("retVal = ",retVal)
        retVal = np.sum(compute_cost)
        print (retVal)
        return retVal

    def equality_constraint(x):
        # print("equality_constraint")
        return np.sum(x.reshape(-1, processors), axis=1) - 1

    constraints = ({'type': 'eq', 'fun': equality_constraint})
    result = minimize(objective, assignment_matrix, constraints=constraints)

    optimal_assignment = np.reshape(result.x, assignment_matrix.shape)
    print ("done")
    return optimal_assignment