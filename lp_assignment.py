'''
lp_assignment.py
The lp solve funtion will take in the number of samples, the number of intervals t, the number of processors, and workload matrix.
It returns a list of list of floats. The length of the list is the number of samples, and the length of each list is the number of processors.
The lp assignment function will take in the list of list of floats, assign each sample a processor, and return a list of integers.
Each cell in the list represents the processor that the sample is assigned to.
'''
import random
from ortools.linear_solver import pywraplp
from typing import List, Tuple

def lp_solve(matrix: list, samples: int, intervals: int, processors: int) -> List[List[float]]:
    # create solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise RuntimeError('Could not create solver')
    # set time limit
    # solver.SetTimeLimit(timeout)
    # assignment variables
    x = {}
    for i in range(samples):
        for k in range(processors):
            x[i, k] = solver.NumVar(0, 1, 'x_{}_{}'.format(i, k))
    # interval time variables
    t = {}
    for j in range(intervals):
        t[j] = solver.NumVar(0, solver.infinity(), 't_{}'.format(j))
    print('Number of variables =', solver.NumVariables())
    # assignment constraints
    for i in range(samples):
        solver.Add(sum(x[i, k] for k in range(processors)) == 1)
    # interval time constraints
    for j in range(intervals):
        for k in range(processors):
            solver.Add(sum(matrix[i][j] * x[i, k] for i in range(samples)) <= t[j])
    print('Number of constraints =', solver.NumConstraints())
    # objective
    solver.Minimize(sum(t[j] for j in range(intervals)))
    solver_optimal = solver.Solve()
    # solve
    print('Objective value =', solver.Objective().Value())
    print('Is optimal solution?', solver_optimal == pywraplp.Solver.OPTIMAL)
    print('Problem solved in {} ms.'.format(solver.wall_time()))
    solution = []
    for i in range(samples):
        tmp = []
        for k in range(processors):
            tmp.append(x[i, k].solution_value())
        solution.append(tmp)
    return solution

# convert lp assignment to assignment by max
def lp_max(solution: list, samples: int, processors: int) -> list:
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
def lp_random(solution: list, samples: int, processors: int) -> list:
    result = []
    for i in range(samples):
        weights = [solution[i][k] for k in range(processors)]
        result.append(random.choices(range(processors), weights = weights)[0])
    return result