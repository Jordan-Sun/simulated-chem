from scipy.optimize import minimize
import numpy as np
from complex_assignment import get_neighbor

# This lp solver will consider both the communication and computation time and try to minimize the total time
# variables:
# samples: s
# intervals: t
# processors: p
# assignment_matrix (s x p)
# workload_matrix (t x s=)
# compute_cost_matrix (t x 1)
# communicate_cost_matrix (t x 1)

def lp_communicate_compute(workload_matrix: list, samples: int, intervals: int, processors: int) -> List[List[float]]:
    
    def objective(assignment_matrix):
        compute_time = workload_matrix @ assignment_matrix

        communicate_time = 0
        for p in processos:
            p_communicate_time = 0
            for index in range(assignment_matrix[:,p]):
                
                neighbor_list = get_neighbor(range,)
                for neighbor in neighbor_list:
                    p_communicate_time += (sample-assignment_matrix[neighbor])
            if p_communicate_time>communicate_time:
                communicate_time = p_communicate_time
        
        return np.sum(np.max(compute_time, axis=1).reshape(-1,1)) + communicate_time * intervals

    def equality_constraint():
        row_sums = np.sum(x, axis=1)
        return row_sums - 1

    assignment_matrix = np.zeros(samples, processors)

    constraints = ({'type': 'eq', 'fun': equality_constraint})

    result = minimize(objective, assignment_matrix.flatten(), constraints=constraints)

    optimal_assignment = np.reshape(result.x, assignment_matrix.shape)

    return assignment_matrix