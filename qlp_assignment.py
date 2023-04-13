'''
qlp_assignment.py
The qlp solve funtion will take in the width x and height y of the sample grid, the number of intervals t, the number of processors, and workload matrix.
It returns a list of list of floats. The length of the list is the number of samples, x * y, and the length of each list is the number of processors.
The qlp assignment function will take in the list of list of floats, assign each sample a processor, and return a list of integers.
Each cell in the list represents the processor that the sample is assigned to.
'''
import random
from ortools.linear_solver import pywraplp
from typing import List, Tuple

# Convert a 2d index to a 1d index with wrapping
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

# Convert a 1d index to a 2d index
def itowh(i: int, width: int, height: int) -> Tuple[int, int]:
    if i < 0 or i >= width * height:
        raise ValueError('i out of bounds')
    return i % width, int(i / width)

def qlp_solve(matrix: list, width: int, height: int, intervals: int, processors: int, comm_cost: float = 0.01) -> List[List[float]]:
    # number of samples
    samples = width * height
    # create solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise RuntimeError('Could not create solver')
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
    objective = sum(t[j] for j in range(intervals))
    communications = 0
    for i in range(samples):
        # we can only add communication cost to the left and up here without loss of generality because of symmetry
        # this allows us to reduce the amount of computations
        m, n = itowh(i, width, height)
        # left
        left = whtoi(m - 1, n, width, height)
        # up
        up = whtoi(m, n - 1, width, height)
        # add communication cost
        for k in range(processors):
            communications += (x[i, k] - x[left, k]) * (x[i, k] - x[left, k])
            communications += (x[i, k] - x[up, k]) * (x[i, k] - x[up, k])
    objective += comm_cost * communications
    solver.Minimize(objective)
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
def qlp_max(solution: list, samples: int, processors: int) -> list:
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
def qlp_random(solution: list, samples: int, processors: int) -> list:
    result = []
    for i in range(samples):
        weights = [solution[i][k] for k in range(processors)]
        result.append(random.choices(range(processors), weights = weights)[0])
    return result