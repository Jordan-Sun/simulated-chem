from scipy.optimize import minimize
import numpy as np
from complex_assignment import get_neighbor
from complex_assignment import whtoi
from complex_assignment import itowh
# This lp solver will consider both the communication and computation time and try to minimize the total time
# variables:
# samples: s
# intervals: t
# processors: p
# assignment_matrix (s x p)
# workload_matrix (t x s=)
# compute_cost_matrix (t x 1)
# communicate_cost_matrix (t x 1)

def lp_communicate_compute(communication_cost: int, width: int, height: int, workload_matrix: list, samples: int, intervals: int, processors: int) -> List[List[float]]:

    def objective(assignment_matrix):
        compute_time = workload_matrix @ assignment_matrix
        compute_cost = np.max(compute_time, axis=1).reshape(compute_cost, (-1, 1))

        communicate_time = np.empty(processors)
        for p in range(processors):
            # PSEUDO-CODE
            # task_array = get_task(p)
            # for task in task_array:
            #   neighbor_array = get_neighbor(task)
            #   for neighbor in neighbor_array:
            #     if neighbor not in task_array:
            #       communicate_time[p] += a+b (communicate)
            # get the indices of the tasks of processor p
            tasks_array = np.nonzero(assignment_matrix[:, p])[0]
            for task in tasks_array:
                neighbor_list = get_neighbor(task, width, height)
                for neighbor_index in neighbor_list:
                    if neighbor_index not in tasks_array:
                        communicate_time += communication_cost

        communicate_cost = np.max(communicate_time)
        return np.sum(compute_cost) + communicate_cost * intervals

    def equality_constraint():
        row_sums = np.sum(x, axis=1)
        return row_sums - 1
    
    assignment_matrix = np.eye(processors)[np.random.choice(processors, samples)]

    constraints = ({'type': 'eq', 'fun': equality_constraint})

    result = minimize(objective, assignment_matrix.flatten(), constraints=constraints)

    optimal_assignment = np.reshape(result.x, assignment_matrix.shape)

    return assignment_matrix