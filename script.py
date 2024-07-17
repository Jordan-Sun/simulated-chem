'''
script.py <task> <function> <job index> [extra]
Runs the function with the parameters specified by the job index.
'''
import lp_assignment
import simple_assignment
import qlp_assignment
from genetic import genetic_algorithm
# import lp_computation_commute_two
import simulation
import animation
import genetic
import sys
import os
import pandas as pd
from typing import Tuple
import complex_assignment
import numpy as np
import math

procs = [16, 128, 1024]
xs = [20, 50, 100, 200]
ts = [20, 50, 100, 200]
rs = [2, 5, 10, 20]
num_trials = 20

def itowh(i: int, width: int, height: int) -> Tuple[int, int]:
    if i < 0 or i >= width * height:
        raise ValueError('i out of bounds')
    return i % width, int(i / width)

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

def reshape_to_2d(matrix_3d):
    height, width, interval = len(matrix_3d), len(matrix_3d[0]), len(matrix_3d[0][0])
    matrix_2d = [matrix_3d[i//width][i%width] for i in range(height*width)]
    return matrix_2d

def coarsen(matrix_2d, width, height, interval,coarse_constant):
    # Calculate the coarsened dimensions
    coarse_width = math.ceil(width/coarse_constant)
    coarse_height = math.ceil(height/coarse_constant)
    coarse_matrix = [[[0]*interval for _ in range(coarse_height)] for _ in range(coarse_width)]
    new_dictionary = {}
    for i in range(coarse_width):
        for j in range(coarse_height):
            tuple_key = (i,j)
            index = whtoi(i, j, coarse_width, coarse_height)
            new_dictionary[index] = []

    for y in range(height):
        j = math.floor(y/coarse_constant)
        for x in range(width):
            i = math.floor(x/coarse_constant)
            tuple_value = whtoi(x, y, width, height)
            new_dictionary[whtoi(i, j, coarse_width, coarse_height)].append(tuple_value)
            for k in range(interval):
                coarse_matrix[i][j][k] += matrix_2d[whtoi(x, y, width, height)][k]
    # Convert coarsened 3D matrix back to 2D
    coarse_matrix_2d = reshape_to_2d(coarse_matrix)
    return coarse_matrix_2d, coarse_width, coarse_height, new_dictionary

# converts the job index to the parameters
# the parameters are xlp_co, y, t, r, p, trial
def index_to_params(index: int) -> Tuple[int, int, int, int, int, int]:
    num = index - 1
    num, trial = divmod(num, num_trials)
    num, index = divmod(num, len(procs))
    p = procs[index]
    num, index = divmod(num, len(rs))
    r = rs[index]
    num, index = divmod(num, len(ts))
    t = ts[index]
    num, index = divmod(num, len(xs))
    x = xs[index]
    return x, x, t, r, p, (trial+1)

# executes a simple assignment function and write the results to a file
def simple(function: str, workdir: str, width: int, height: int, p: int, extra: int =  -1):
    # the directory to write the results to
    outdir = os.path.join(workdir, 'p_{}'.format(p))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # the name of the file to write the assignments to
    if extra >= 0:
        file_name = os.path.join(outdir, '{}_{}.assignment'.format(function, extra))
    else:
        file_name = os.path.join(outdir, '{}.assignment'.format(function))
    # if the file already exists, do not run the assignment function
    if os.path.exists(file_name):
        print('Skipping assignment function at {}'.format(file_name))
        return 1
    else:
        print('Running assignment function at {}'.format(file_name))
    # the assignment function
    assignments = None
    if function == 'proximity_y':
        assignments = simple_assignment.proximity_y(width, height, p)
    elif function == 'proximity_x':
        assignments = simple_assignment.proximity_x(width, height, p)
    elif function == 'proximity_matrix':
        assignments = simple_assignment.proximity_matrix(width, height, p)
    elif function == 'simple_random':
        assignments = simple_assignment.simple_random(width, height, p)
    elif function == 'balanced_random':
        assignments = simple_assignment.balanced_random(width, height, p)
    else:
        print('Invalid assignment function')
        return -6
    # write the assignments to a file as csv
    array = np.array(assignments).reshape((width,height))
    df = pd.DataFrame(array)
    df.to_csv(file_name, header=False, index=False)
    return 0

# executes a complicated assignment function and write the results to a file
def complicated(function: str, workdir: str, width: int, height: int, t: int, p: int, extra: int = -1, send_cost: int = 1, receive_cost: int = 1, tuning_constant = 1, coarse: int = 0, fine_tune: bool = 0):
    print("width: ",width)
    print("height: ",height)
    print("interval: ",t)
    print("fine_tune: ",fine_tune)
    # the directory to write the results to
    outdir = os.path.join(workdir, 'p_{}'.format(p))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # the name of the file to write the assignments to
    if extra >= 0:
        file_name = os.path.join(outdir, '{}_{}.assignment'.format(function, extra))
    else:
        file_name = os.path.join(outdir, '{}.assignment'.format(function))
    if send_cost != 1 and function != 'greedy':
        file_name = file_name.replace('.assignment', '_send{}.assignment'.format(send_cost))
    if receive_cost != 1 and function != 'greedy':
        file_name = file_name.replace('.assignment', '_recv{}.assignment'.format(receive_cost))
    if tuning_constant != 1 and function != 'greedy':
        file_name = file_name.replace('.assignment', '_tuning{}.assignment'.format(tuning_constant))
    if coarse != 0:
        file_name = file_name.replace('.assignment', '_coarse{}.assignment'.format(coarse))
    if fine_tune != False:
        file_name = file_name.replace('.assignment', '_finetune{}.assignment'.format(fine_tune))
    print(file_name)
    # if the file already exists, do not run the assignment function
    if os.path.exists(file_name):
        print('Skipping assignment function at {}'.format(file_name))
        return 1
    else:
        print('Running assignment function at {}'.format(file_name))
        # the name of the file to read the workload from
        input_name = os.path.join(workdir, 'workload.csv')
        if not os.path.exists(input_name):
            print('Workload file does not exist')
            return -4
        # read the workload from the file
        workload = pd.read_csv(input_name, header='infer', index_col=0).values.tolist()
        samples = width * height
        if len(workload) != samples:
            print('Workload file has invalid number of samples')
            return -5
        if coarse != 0 and function != 'mosaic_greedy':
            # print("original workload file dimension: ")
            # print("samples: ",len(workload))
            # print("interval: ", len(workload[0]))
            # print("original workload: \n", workload)
            workload, width, height, new_dictionary = coarsen(workload, width, height, t, coarse)
            # print("new workload file dimension: ")
            # print("original workload file dimension: ")
            # print("samples: ",len(workload))
            # print("interval: ", len(workload[0]))
            # print("coarsed workload: \n", workload)
            send_cost *= coarse
            receive_cost *= coarse 

    # the assignment function
    assignments = None
    if function == 'greedy':
        assignments = complex_assignment.greedy(workload, width, height, t, p, extra)
    elif function == 'reassign_greedy':
        # Read the 2d assignment, flatten it, and pass it to the reassign_greedy function as a list
        original_assignments = pd.read_csv(os.path.join(outdir, 'original.assignment'), header=None).values.flatten().tolist()
        assignments = complex_assignment.reassign_greedy(workload, width, height, t, p, original_assignments, extra)
    elif function == 'balance_greedy':
        # Read the 2d assignment, flatten it, and pass it to the balance_greedy function as a list
        original_assignments = pd.read_csv(os.path.join(outdir, 'original.assignment'), header=None).values.flatten().tolist()
        assignments = complex_assignment.balance_greedy(workload, width, height, t, p, original_assignments, extra)
    elif function == 'reshuffle_greedy':
        # Read the 2d assignment, flatten it, and pass it to the balance_greedy function as a list
        original_assignments = pd.read_csv(os.path.join(outdir, 'original.assignment'), header=None).values.flatten().tolist()
        assignments = complex_assignment.reshuffle_greedy(workload, width, height, t, p, original_assignments, extra)
    elif function == 'greedy_independent_broadcast':
        assignments = complex_assignment.greedy_with_communication(workload, width, height, t, p, extra, send_cost, receive_cost, complex_assignment.independent_broadcast_cost_function)
    elif function == 'greedy_dependent_broadcast':
        assignments = complex_assignment.greedy_with_communication(workload, width, height, t, p, extra, send_cost, receive_cost, complex_assignment.dependent_broadcast_cost_function)
    elif function == 'greedy_independent_unicast':
        assignments = complex_assignment.greedy_with_communication(workload, width, height, t, p, extra, send_cost, receive_cost, complex_assignment.independent_unicast_cost_function)
    elif function == 'greedy_dependent_unicast':
        assignments = complex_assignment.greedy_with_communication(workload, width, height, t, p, extra, send_cost, receive_cost, complex_assignment.dependent_unicast_cost_function)
    elif function == 'greedy_weak_neighbor_dependent_unicast':
        assignments = complex_assignment.greedy_with_communication(workload, width, height, t, p, extra, send_cost, receive_cost, complex_assignment.weak_neighbor_dependent_unicast_cost_function)
    elif function == 'greedy_prioritize_communication':
        assignments = complex_assignment.greedy_prioritize_communication(workload, width, height, t, p, extra, send_cost, receive_cost, tuning_constant)
    elif function == 'greedy_prioritize_communication_fine_tune':
        assignments = complex_assignment.greedy_prioritize_communication_fine_tune(workload, width, height, t, p, extra, send_cost, receive_cost, tuning_constant, coarse, fine_tune)
    elif function == 'mosaic_greedy':
        assignments = complex_assignment.mosaic_greedy(workload, width, height, t, p, extra, tuning_constant, coarse)
    elif function == 'blocked_greedy':
        assignments = complex_assignment.blocked_greedy(workload, width, height, t, p, extra, tuning_constant, coarse)
    else:
        print('Invalid assignment function')
        return -6
    # write the assignments to a file as csv
    array = np.array(assignments).reshape((width,height))
    df = pd.DataFrame(array)
    df.to_csv(file_name, header=False, index=False)
    return 0

# executes a lp assignment function and write the results to a file
def lp(function: str, workdir: str, width: int, height: int, t: int, p: int, extra: int = -1):
    # the directory to write the results to
    outdir = os.path.join(workdir, 'p_{}'.format(p))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if function == 'solve':
        # the name of the file to write the assignments to
        file_name = os.path.join(outdir, 'lp.solution')
        # if the file already exists, do not run the lp function
        if os.path.exists(file_name):
            print('Skipping lp function at {}'.format(file_name))
            return 1
        else:
            print('Running lp function at {}'.format(file_name))
        # the name of the file to read the workload from
        input_name = os.path.join(workdir, 'workload.csv')
        if not os.path.exists(input_name):
            print('Workload file does not exist')
            return -4
        # read the workload from the file
        workload = pd.read_csv(input_name, header='infer', index_col=0).values.tolist()
        samples = width * height
        if len(workload) != samples:
            print('Workload file has invalid number of samples')
            return -5
        solution = lp_assignment.lp_solve(workload, samples, t, p)
        # write the assignments to a file
        df = pd.DataFrame(solution)
        df.to_csv(file_name, header=False, index=False)
        return 0
    elif function == 'verify':
        # the name of the file to write the assignments to
        file_name = os.path.join(outdir, 'lp_verification.result')
        # if the file already exists, do not run the verification function
        if os.path.exists(file_name):
            print('Skipping verification function at {}'.format(file_name))
            return 1
        else:
            print('Running verification function at {}'.format(file_name))
        samples = width * height
        # the name of the file to read the workload from
        input_name = os.path.join(workdir, 'workload.csv')
        if not os.path.exists(input_name):
            print('Workload file does not exist')
            return -4
        # read the workload from the file
        workload = pd.read_csv(input_name, header='infer', index_col=0).values.tolist()
        if len(workload) != samples:
            print('Workload file has invalid number of samples')
            return -5
        # the name of the file to read the solution from
        input_name = os.path.join(outdir, 'lp.solution')
        if not os.path.exists(input_name):
            print('Solution file does not exist')
            return -4
        # read the solution from the file
        solution = pd.read_csv(input_name, header=None).values.tolist()
        if len(solution) != samples:
            print('Solution file has invalid number of samples')
            return -5
        # verify the computation
        result = simulation.verify(workload, solution, width, height, t, p)
        # write the result to a file
        with open(file_name, 'w') as file:
            file.write('lp computation, {}\n'.format(result))
    else:
        # the name of the file to read the solution from
        input_name = os.path.join(outdir, 'lp.solution')
        if not os.path.exists(input_name):
            print('Solution file does not exist')
            return -4
        # read the solution from the file
        solution = pd.read_csv(input_name, header=None).values.tolist()
        samples = width * height
        if len(solution) != samples:
            print('Solution file has invalid number of samples')
            return -5
        # the name of the file to write the assignments to
        if extra >= 0:
            file_name = os.path.join(outdir, 'lp_{}_{}.assignment'.format(function, extra))
        else:
            file_name = os.path.join(outdir, 'lp_{}.assignment'.format(function))
        # if the file already exists, do not run the post processing function
        if os.path.exists(file_name):
            print('Skipping post processing function at {}'.format(file_name))
            return 1
        else:
            print('Running post processing function at {}'.format(file_name))
        # the assignment function
        assignments = None
        if function == 'max':
            assignments = lp_assignment.lp_max(solution, samples, p)
        elif function == 'random':
            assignments = lp_assignment.lp_random(solution, samples, p)
        else:
            print('Invalid assignment function')
            return -6
        # write the assignments to a file
        array = np.array(assignments).reshape((width,height))
        df = pd.DataFrame(array)
        df.to_csv(file_name, header=False, index=False)
    return 0

# executes a qlp assignment function and write the results to a file
def qlp(function: str, workdir: str, width: int, height: int, t: int, p: int, extra: int = -1):
    # the directory to write the results to
    outdir = os.path.join(workdir, 'p_{}'.format(p))
    solution_file = 'qlp.solution'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if function == 'solve':
        # the name of the file to write the assignments to
        file_name = os.path.join(outdir, solution_file)
        # if the file already exists, do not run the lp function
        if os.path.exists(file_name):
            print('Skipping qlp function at {}'.format(file_name))
            return 1
        else:
            print('Running qlp function at {}'.format(file_name))
        # the name of the file to read the workload from
        input_name = os.path.join(workdir, 'workload.csv')
        if not os.path.exists(input_name):
            print('Workload file does not exist')
            return -4
        # read the workload from the file
        workload = pd.read_csv(input_name, header='infer', index_col=0).values.tolist()
        samples = width * height
        if len(workload) != samples:
            print('Workload file has invalid number of samples')
            return -5
        solution = qlp_assignment.qlp_solve(workload, samples, t, p)
        # write the assignments to a file
        df = pd.DataFrame(solution)
        df.to_csv(file_name, header=False, index=False)
        return 0
    else:
        # the name of the file to read the solution from
        input_name = os.path.join(outdir, solution_file)
        if not os.path.exists(input_name):
            print('Solution file does not exist')
            return -4
        # read the solution from the file
        solution = pd.read_csv(input_name, header=None).values.tolist()
        samples = width * height
        if len(solution) != samples:
            print('Solution file has invalid number of samples')
            return -5
        # the name of the file to write the assignments to
        if extra >= 0:
            file_name = os.path.join(outdir, 'qlp_{}_{}.assignment'.format(function, extra))
        else:
            file_name = os.path.join(outdir, 'qlp_{}.assignment'.format(function))
        # if the file already exists, do not run the post processing function
        if os.path.exists(file_name):
            print('Skipping post processing function at {}'.format(file_name))
            return 1
        else:
            print('Running post processing function at {}'.format(file_name))
        # the assignment function
        assignments = None
        if function == 'max':
            assignments = qlp_assignment.qlp_max(solution, samples, p)
        elif function == 'random':
            assignments = qlp_assignment.qlp_random(solution, samples, p)
        else:
            print('Invalid assignment function')
            return -6
        # write the assignments to a file
        array = np.array(assignments).reshape((width,height))
        df = pd.DataFrame(array)
        df.to_csv(file_name, header=False, index=False)
    return 0

def genetic (function: str, workdir: str, sendCost: int, recvCost: int, width: int, height: int, t: int, p: int, extra: int = -1):
    outdir = os.path.join(workdir, 'p_{}'.format(p))
    function = 'genetic'
    solution_file = 'genetic.solution'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # the name of the file to write the assignments to
    if extra >= 0:
        file_name = os.path.join(outdir, '{}_{}.assignment'.format(function, extra))
    else:
        file_name = os.path.join(outdir, '{}.assignment'.format(function))
    if sendCost != 1:
        file_name = file_name.replace('.assignment', '_send{}.assignment'.format(sendCost))
    if recvCost != 1:
        file_name = file_name.replace('.assignment', '_recv{}.assignment'.format(recvCost))
    # if the file already exists, do not run the assignment function
    if os.path.exists(file_name):
        print('Skipping assignment function at {}'.format(file_name))
        return 1
    else:
        print('Running assignment function at {}'.format(file_name))
    # the name of the file to read the workload from
        input_name = os.path.join(workdir, 'workload.csv')
        if not os.path.exists(input_name):
            print('Workload file does not exist')
            return -4
        # read the workload from the file
        workload = pd.read_csv(input_name, header='infer', index_col=0).values.tolist()
        samples = width * height
        if len(workload) != samples:
            print('Workload file has invalid number of samples')
            return -5
    # the assignment function
    assignments = genetic_algorithm(workload, width, height, t, p, extra, sendCost, recvCost)
    # write the assignments to a file
    array = np.array(assignments).reshape((width,height))
    df = pd.DataFrame(array)
    df.to_csv(file_name, header=False, index=False)
    return 0

def lp_compute_commute(function: str, workdir: str, sendCost: int, recvCost: int, width: int, height: int, t: int, p: int, extra: int = -1):
###########
    samples = width * height
    print("extra ", extra)
    # the directory to write the results to
    outdir = os.path.join(workdir, 'p_{}'.format(p))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    solution_file = 'lp_compute_commute.solution'
    # the name of the solution file to write the assignments to
    if sendCost != 1:
        solution_file = solution_file.replace('.solution', '_send{}.solution'.format(sendCost))
    if recvCost != 1:
        solution_file = solution_file.replace('.solution', '_recv{}.solution'.format(recvCost))
###########
    
    if function == 'solve':
        # initialize an assignment to feed into the lp function
        matrix = np.zeros((samples,p))
        for i in range(samples):
            matrix[i][np.random.randint(0, p)] =  1
        # the name of the file to write the solution to
        file_name = os.path.join(outdir, solution_file)
        # if the file already exists, do not run the lp function
        if os.path.exists(file_name):
            print('Skipping lp_compute_commute function at {}'.format(file_name))
            return 1
        else:
            print('Running lp_compute_commute function at {}'.format(file_name))
        # the name of the file to read the workload from
        input_name = os.path.join(workdir, 'workload.csv')
        if not os.path.exists(input_name):
            print('Workload file does not exist')
            return -4
        # read the workload from the file
        workload = pd.read_csv(input_name, header='infer', index_col=0).values.tolist()
        samples = width * height
        if len(workload) != samples:
            print('Workload file has invalid number of samples')
            return -5
        solution = lp_computation_commute_two.lp_compute_commute(matrix, sendCost, recvCost, width, height, workload, samples, t, p)
        # write the assignments to a file
        df = pd.DataFrame(solution)
        df.to_csv(file_name, header=False, index=False)
        return 0
    else:
        # the name of the file to read the solution from
        input_name = os.path.join(outdir, solution_file)
        if not os.path.exists(input_name):
            print('Solution file does not exist')
            return -4
        # read the solution from the file
        solution = pd.read_csv(input_name, header=None).values.tolist()
        samples = width * height
        if len(solution) != samples:
            print('Solution file has invalid number of samples')
            return -5
        # the name of the file to write the assignments to
        if extra >= 0:
            file_name = os.path.join(outdir, 'lp_compute_commute_{}_{}.assignment'.format(function, extra))
        else:
            file_name = os.path.join(outdir, 'lp_compute_commute_{}.assignment'.format(function))
        if sendCost != 1:
            file_name = file_name.replace('.assignment', '_send{}.assignment'.format(sendCost))
        if recvCost != 1:
            file_name = file_name.replace('.assignment', '_recv{}.assignment'.format(recvCost))
        # if the file already exists, do not run the post processing function
        print("file name: ",file_name)
        if os.path.exists(file_name):
            print('Skipping post processing function at {}'.format(file_name))
            return 1
        else:
            print('Running post processing function at {}'.format(file_name))
        # the assignment function
        assignments = None
        if function == 'max':
            assignments = lp_computation_commute_two.lp_compute_commute_max(solution, samples, p)
        elif function == 'random':
            assignments = lp_computation_commute_two.lp_compute_commute_random(solution, samples, p)
        else:
            print('Invalid assignment function')
            return -6
        # write the assignments to a file
        with open(file_name, 'w') as file:
            for y in range(height):
                file.write(','.join([str(assignments[simple_assignment.cord_to_index(width, height, x, y)]) for x in range(width)]) + '\n')
    return 0

# executes a simulation function and write the results to a file
def simulate(function: str, workdir: str, width: int, height: int, t: int, p: int, extra: int, sendCost: int, recvCost: int, tuning_constant = 1, coarse: int = 0, fine_tune: bool = False):
    # the directory to write the results to
    outdir = os.path.join(workdir, 'p_{}'.format(p))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # the name of the file to read the workload from
    input_name = os.path.join(workdir, 'workload.csv')
    if not os.path.exists(input_name):
        print('Workload file does not exist')
        return -4
    # read the workload from the file
    workload = pd.read_csv(input_name, header='infer', index_col=0).values.tolist()
    samples = width * height
    if len(workload) != samples:
        print('Workload file has invalid number of samples')
        return -5
    if coarse != 0 and function != 'mosaic_greedy': #TODO: Is this correct?
        # print("original workload file dimension: ")
        # print("samples: ",len(workload))
        # print("interval: ", len(workload[0]))
        workload, width, height, new_dictionary = coarsen(workload, width, height, t, coarse)
        # print("new workload file dimension: ")
        # print("samples: ",len(workload))
        # print("interval: ", len(workload[0]))
        samples = width * height
        
    if function == 'bound':
        # the name of the file to write the assignments to
        file_name = os.path.join(outdir, 'bound.result')
        # if the file already exists, do not run the bound function
        if os.path.exists(file_name):
            print('Skipping bound function at {}'.format(file_name))
            return 1
        else:
            print('Running bound function at {}'.format(file_name))
        # calculate the bound
        try:
            span, work = simulation.workspan(workload, samples, t)
        except ValueError as e:
            print(e)
            return -7
        result = simulation.bound(span, work, p)
        # write the result to a file
        with open(file_name, 'w') as file:
            file.write('bound, {}\n'.format(result))
    else:
        # the name of the assignment file and output file
        if extra >= 0:
            assignment_name = os.path.join(outdir, '{}_{}.assignment'.format(function, extra))
            output_name = os.path.join(outdir, '{}_{}.result'.format(function, extra))
        else:
            assignment_name = os.path.join(outdir, '{}.assignment'.format(function))
            output_name = os.path.join(outdir, '{}.result'.format(function))
        communication_aware_functions = ['greedy_independent_broadcast', 'greedy_dependent_broadcast', 'greedy_independent_unicast', 'greedy_dependent_unicast', 'greedy_weak_neighbor_dependent_unicast', 'lp_compute_commute_max', 'lp_compute_commute_random', 'greedy_prioritize_communication', 'greedy_prioritize_communication_fine_tune']
        
        if function in communication_aware_functions:
            if sendCost != 1:
                assignment_name = assignment_name.replace('.assignment', '_send{}.assignment'.format(sendCost))
            if recvCost != 1:
                assignment_name = assignment_name.replace('.assignment', '_recv{}.assignment'.format(recvCost))
        if sendCost != 1:
            output_name = output_name.replace('.result', '_send{}.result'.format(sendCost))
        if recvCost != 1:
            output_name = output_name.replace('.result', '_recv{}.result'.format(recvCost))
        if tuning_constant != 1:
            output_name = output_name.replace('.result', '_tuning{}.result'.format(tuning_constant))
            assignment_name = assignment_name.replace('.assignment', '_tuning{}.assignment'.format(tuning_constant))
        if coarse != 0:
            output_name = output_name.replace('.result', '_coarse{}.result'.format(coarse))
            assignment_name = assignment_name.replace('.assignment', '_coarse{}.assignment'.format(coarse))
        if fine_tune != False:
            output_name = output_name.replace('.result', '_finetune{}.result'.format(fine_tune))
            assignment_name = assignment_name.replace('.assignment', '_finetune{}.assignment'.format(fine_tune))
        print("assignment_name: ", assignment_name)
        if not os.path.exists(assignment_name):
            print(assignment_name)
            print('Assignment file does not exist')
            return -6
        # if the file already exists, do not run the simulation function
        if os.path.exists(output_name):
            print('Skipping simulation function at {}'.format(output_name))
            return 1
        else:
            print('Running simulation function at {}'.format(output_name))
        # read the assignments from the file
        assignments_matrix = pd.read_csv(assignment_name, header=None).values.tolist()
        # convert the matrix to a list
        assignments = [cell for row in assignments_matrix for cell in row]
        if len(assignments) != samples:
            print('Assignment file has invalid number of samples')
            return -7
        # simulate the workload
        if coarse != 0 and function != 'mosaic_greedy': #TODO: Is this line correct?
            sendCost = sendCost * coarse
            recvCost = recvCost * coarse 
        try:
            computation, broadcast, unicast, computationBroadcast, computationUnicast = simulation.simulate(workload, assignments, width, height, t, p, sendCost, recvCost)
        except ValueError as e:
            print(e)
            return -7
        # write the results to a file
        with open(output_name, 'w') as file:
            file.write('Computation, {}\n'.format(computation))
            file.write('Broadcast, {}\n'.format(broadcast))
            file.write('Unicast, {}\n'.format(unicast))
            # file.write('ComputationBroadcast, {}\n'.format(computationBroadcast))
            # file.write('ComputationUnicast, {}\n'.format(computationUnicast))
    return 0

# executes the alternative simulation function and write the results to a file
def alt_simulate(function: str, workdir: str, width: int, height: int, t: int, p: int, extra: int, sendCost: int, recvCost: int):
    # the directory to write the results to
    # print("function: ", function)
    outdir = os.path.join(workdir, 'p_{}'.format(p))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # the name of the file to read the workload from
    input_name = os.path.join(workdir, 'workload.csv')
    if not os.path.exists(input_name):
        print('Workload file does not exist')
        return -4
    # read the workload from the file
    workload = pd.read_csv(input_name, header='infer', index_col=0).values.tolist()
    samples = width * height
    if len(workload) != samples:
        print('Workload file has invalid number of samples')
        return -5

    # the name of the assignment file and output file
    if extra >= 0:
        assignment_name = os.path.join(outdir, '{}_{}.assignment'.format(function, extra))
        output_name = os.path.join(outdir, '{}_{}.altres'.format(function, extra))
    else:
        assignment_name = os.path.join(outdir, '{}.assignment'.format(function))
        output_name = os.path.join(outdir, '{}.altres'.format(function))
    
    communication_aware_functions = ['greedy_independent_broadcast', 'greedy_dependent_broadcast', 'greedy_independent_unicast', 'greedy_dependent_unicast', 'greedy_weak_neighbor_dependent_unicast', 'lp_compute_commute']
    if function in communication_aware_functions:
        if sendCost != 1:
            assignment_name = assignment_name.replace('.assignment', '_send{}.assignment'.format(sendCost))
        if recvCost != 1:
            assignment_name = assignment_name.replace('.assignment', '_recv{}.assignment'.format(recvCost))
    if sendCost != 1:
        output_name = output_name.replace('.altres', '_send{}.altres'.format(sendCost))
    if recvCost != 1:
        output_name = output_name.replace('.altres', '_recv{}.altres'.format(recvCost))
    
    if not os.path.exists(assignment_name):
        print('Assignment file does not exist')
        return -6
    # if the file already exists, do not run the simulation function
    if os.path.exists(output_name):
        print('Skipping simulation function at {}'.format(output_name))
        return 1
    else:
        print('Running simulation function at {}'.format(output_name))
    # read the assignments from the file
    assignments_matrix = pd.read_csv(assignment_name, header=None).values.tolist()
    # convert the matrix to a list
    assignments = [cell for row in assignments_matrix for cell in row]
    if len(assignments) != samples:
        print('Assignment file has invalid number of samples')
        return -7
    # simulate the workload
    try:
        computation, unicast = simulation.alt_simulate(workload, assignments, width, height, t, p, sendCost, recvCost)
    except ValueError as e:
        print(e)
        return -7
    # write the results to a file
    with open(output_name, 'w') as file:
        file.write('Computation, {}\n'.format(computation))
        file.write('Unicast, {}\n'.format(unicast))
    return 0

def animate(function: str, workdir: str, width: int, height: int, t: int, p: int, extra: int, sendCost: int, recvCost: int):
    # the directory to write the results to
    outdir = os.path.join(workdir, 'p_{}'.format(p))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # the name of the file to read the workload from
    input_name = os.path.join(workdir, 'workload.csv')
    if not os.path.exists(input_name):
        print('Workload file does not exist')
        return -4
    # read the workload from the file
    workload = pd.read_csv(input_name, header='infer', index_col=0).values.tolist()
    samples = width * height
    if len(workload) != samples:
        print('Workload file has invalid number of samples')
        return -5

    # the name of the assignment file and output file
    if extra >= 0:
        assignment_name = os.path.join(outdir, '{}_{}.assignment'.format(function, extra))
        output_name = os.path.join(outdir, '{}_{}.gif'.format(function, extra))
    else:
        assignment_name = os.path.join(outdir, '{}.assignment'.format(function))
        output_name = os.path.join(outdir, '{}.gif'.format(function))
    
    communication_aware_functions = ['greedy_independent_broadcast', 'greedy_dependent_broadcast', 'greedy_independent_unicast', 'greedy_dependent_unicast', 'greedy_weak_neighbor_dependent_unicast', 'lp_compute_commute']
    if function in communication_aware_functions:
        if sendCost != 1:
            assignment_name = assignment_name.replace('.assignment', '_send{}.assignment'.format(sendCost))
        if recvCost != 1:
            assignment_name = assignment_name.replace('.assignment', '_recv{}.assignment'.format(recvCost))
    if sendCost != 1:
        output_name = output_name.replace('.gif', '_send{}.gif'.format(sendCost))
    if recvCost != 1:
        output_name = output_name.replace('.gif', '_recv{}.gif'.format(recvCost))
    
    if not os.path.exists(assignment_name):
        print('Assignment file does not exist')
        return -6
    # if the file already exists, do not run the simulation function
    if os.path.exists(output_name):
        print('Skipping animation function at {}'.format(output_name))
        return 1
    else:
        print('Running animation function at {}'.format(output_name))
    # read the assignments from the file
    assignments_matrix = pd.read_csv(assignment_name, header=None).values.tolist()
    # convert the matrix to a list
    assignments = [cell for row in assignments_matrix for cell in row]
    if len(assignments) != samples:
        print('Assignment file has invalid number of samples')
        return -7
    # simulate the workload
    try:
        ani = animation.animate(workload, assignments, width, height, t, p)
    except ValueError as e:
        print(e)
        return -7
    # export the animation to a file
    ani.save(output_name)
    return 0

# temporary function to test the dynamic reassignment function
def dynamic(function: str, workdir: str, width: int, height: int, t: int, p: int, reassignment_cost: float = -1, lower_ratio: float = 1, upper_ratio: float = 1):
    # display the parameters
    print("width: ", width)
    print("height: ", height)
    print("interval: ", t)
    print("processors: ", p)
    print("reassignment_cost: ", reassignment_cost)

    # the name of the file to read the workload from
    input_name = os.path.join(workdir, 'workload.csv')
    if not os.path.exists(input_name):
        print('Workload file does not exist')
        return -4
    # read the workload from the file
    workload = pd.read_csv(input_name, header='infer', index_col=0)
    samples = width * height
    if len(workload) != samples:
        print('Workload file has invalid number of samples')
        return -5
    # the directory to read the original assignments from
    workdir = os.path.join(workdir, 'p_{}'.format(p))
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    # read the original assignments from the file
    assignment_name = os.path.join(workdir, 'original.assignment')
    if not os.path.exists(assignment_name):
        print('Assignment file does not exist')
        return -6
    # read the 2d assignment, flatten it, and pass it to the balance_greedy function as a list
    original_assignments = pd.read_csv(assignment_name, header=None).values.flatten().tolist()

    # the directory to write the results to
    outdir = os.path.join(workdir, 'dynamic_{}'.format(reassignment_cost))
    if lower_ratio != 1 or upper_ratio != 1:
        outdir = outdir + '_[{},{}]'.format(lower_ratio, upper_ratio)
    if os.path.exists(outdir):
        print('Skipping dynamic reassignment function at {}'.format(outdir))
        return 1
    else:
        os.mkdir(outdir)
        print('Running dynamic reassignment function at {}'.format(outdir))
    
    # run the assignment function at each interval
    for i in range(t):
        # the name of the file to write the assignments to
        file_name = os.path.join(outdir, 'interval_{}.assignment'.format(i))
        # the assignment function
        assignments = complex_assignment.dynamic_reassignment(workload.iloc[:, i], width, height, p, original_assignments, reassignment_cost, lower_ratio, upper_ratio)
        # write the assignments to a file as csv
        array = np.array(assignments).reshape((width,height))
        df = pd.DataFrame(array)
        df.to_csv(file_name, header=False, index=False)
    return 0

# the assignment function
# the first argument is the name of the task to run
# the second argument is the name of the function to run
# the third argument is the job index
# the fourth argument is an optional extra parameter
# the fifth argument is the send cost
# the sixth argument is the receive cost
def main():
    if len(sys.argv) < 4:
        print('Usage: python3 script.py <task> <function> <index> [extra] [sendCost] [recvCost] [tuning constant] [coarse]')
        return -1
    task_name = sys.argv[1]
    print("task_name: ", task_name)
    function_name = sys.argv[2]
    print("function name: ", function_name)
    index = int(sys.argv[3])
    if len(sys.argv) > 4:
        extra = int(sys.argv[4])
    else:
        extra = -1
    print("extra: ",extra)
    if len(sys.argv) > 5:
        sendCost = float(sys.argv[5])
    else:
        sendCost = 1
    print("send cost: ",sendCost)
    if len(sys.argv) > 6:
        recvCost = float(sys.argv[6])
    else:
        recvCost = 1
    if len(sys.argv) > 7:
        tuning_constant = float(sys.argv[7])
        if tuning_constant > 1 or tuning_constant < 0:
            print("Invalid tuning constant")
            return -2
    else:
        tuning_constant = 1
    print("tuning_constant : ",tuning_constant)
    if len(sys.argv) > 8:
        coarse = int(sys.argv[8])
    else:
        coarse = 0
    print("coarse : ",coarse)
    print("len(sys.argv): ",len(sys.argv))
    if len(sys.argv) > 9:
        fine_tune = True
    else:
        fine_tune = False
    print("fine_tune is: ",fine_tune)
    if index < 1:
        print('Invalid job index')
        return -2
    print("index : ",index)
    # ###### expand folder output/x_5/y_5
    # for base_directory in ['t_20', 't_50', 't_100', 't_200']:
    #     this_directory = os.path.join("output", "x_5", "y_5",base_directory)
    #     for r in ['2','5','10','20']:
    #         r_name = f"r_{r}"
    #         r_path = os.path.join(this_directory, r_name)
    #         os.makedirs(r_path, exist_ok=True)
    #         for x in range(1, 21):
    #             trial_name = f"trial_{x}"
    #             trial_path = os.path.join(r_path, trial_name)
    #             os.makedirs(trial_path, exist_ok=True)
    #             for p in ['5', '16', '128', '1024']: 
    #                 p_name = f"p_{p}"
    #                 p_path = os.path.join(trial_path, p_name)
    #                 os.makedirs(p_path, exist_ok=True)
    # ######

    #####
    # # convert the job index to the parameters
    # x, y, t, r, p, trial = index_to_params(index)
    # workdir = os.path.join('output', 'x_{}'.format(x), 'y_{}'.format(y), 't_{}'.format(t), 'r_{}'.format(r), 'trial_{}'.format(trial))
    #####

    x = 8496
    y = 24
    t = 145
    p = 24
    workdir = os.path.relpath('nc4')

    print('Entering directory: {}'.format(workdir))
    if not os.path.exists(workdir):
        print('Working directory does not exist')
        return -2
    # execute the subtask
    if task_name == 'simple':
        return simple(function_name, workdir, x, y, p, extra)
    elif task_name == 'complicated':
        return complicated(function_name, workdir, x, y, t, p, extra, sendCost, recvCost, tuning_constant, coarse, fine_tune)
    elif task_name == 'lp':
        return lp(function_name, workdir, x, y, t, p, extra)
    elif task_name == 'qlp':
        return qlp(function_name, workdir, x, y, t, p, extra)
    elif task_name == 'simulate':
        return simulate(function_name, workdir, x, y, t, p, extra, sendCost, recvCost, tuning_constant, coarse, fine_tune)
    elif task_name == 'alt_simulate':
        return alt_simulate(function_name, workdir, x, y, t, p, extra, sendCost, recvCost)
    elif task_name == 'lp_compute_commute':
        return lp_compute_commute(function_name, workdir, sendCost, recvCost, x, y, t, p, extra)
    elif task_name == 'genetic_algorithm':
        return genetic(function_name, workdir, sendCost, recvCost, x, y, t, p, extra)
    elif task_name == 'animate':
        return animate(function_name, workdir, x, y, t, p, extra, sendCost, recvCost)
    elif task_name == 'dynamic':
        return dynamic(function_name, workdir, x, y, t, p, extra, sendCost, recvCost)
    else:
        print('Invalid task name')
        return -3

if __name__ == '__main__':
    main()