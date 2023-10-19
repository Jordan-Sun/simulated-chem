import simulation
import numpy as np
import time
import csv
import complex_assignment
import sys

def get_computation_time_interval(workload_matrix, assignment_matrix, width, height, intervals, processors):
    # Validate the input
    samples = width * height
    if samples != len(workload_matrix):
        raise ValueError('Invalid number of samples, expected ' + str(len(workload_matrix)) + ', got ' + str(samples) + '.')
    if intervals != len(workload_matrix[0]):
        raise ValueError('Invalid number of intervals, expected ' + str(len(workload_matrix[0])) + ', got ' + str(intervals) + '.')
    totalComputation = 0
    # print(workload_matrix)
    # print(assignment_matrix)
    # convert assignment matrix into another form
    helper_assignment_matrix = np.zeros((samples, processors))
    for sample in range(samples):
        helper_assignment_matrix[sample][assignment_matrix[sample]] = 1
    compute_time_matrix = helper_assignment_matrix.T @ np.array(workload_matrix)
    # print(helper_assignment_matrix)
    # print(workload_matrix)
    # print(compute_time_matrix)
    max_compute_time = [max(column) for column in zip(*compute_time_matrix)]
    return max_compute_time #, sum(max_compute_time)

def print_assignment(input_assignment, width, height):
    for i in range(height):
        for j in range(width):
            print(input_assignment[j + width * i], end=',')
        print("\n")
    return 0

def print_list(input_list, width, height):
    for i in range(height):
        for j in range(width):
            print(input_list[i][j])
    print("\n")
    return 0

def run_naive_greedy(processors, intervals, width, height, workload_matrix, send_cost, receive_cost):
    extra = -1
    assignment_matrix = complex_assignment.greedy(workload_matrix, width, height, intervals, processors, extra)
    print(assignment_matrix)
    # print_assignment(assignment_matrix, width, height)
    print(len(assignment_matrix))
    return assignment_matrix

def run_greedy_with_communication(processors, intervals, width, height, workload_matrix, send_cost, receive_cost, function):
    extra = -1
    assignment_matrix = None
    if function == 'greedy_independent_broadcast':
        assignment_matrix = complex_assignment.greedy_with_communication(workload_matrix, width, height, intervals, processors, extra, send_cost, receive_cost, complex_assignment.independent_broadcast_cost_function)
    elif function == 'greedy_dependent_broadcast':
        assignment_matrix = complex_assignment.greedy_with_communication(workload_matrix, width, height, intervals, processors, extra, send_cost, receive_cost, complex_assignment.dependent_broadcast_cost_function)
    elif function == 'greedy_independent_unicast':
        assignment_matrix = complex_assignment.greedy_with_communication(workload_matrix, width, height, intervals, processors, extra, send_cost, receive_cost, complex_assignment.independent_unicast_cost_function)
    elif function == 'greedy_dependent_unicast':
        assignment_matrix = complex_assignment.greedy_with_communication(workload_matrix, width, height, intervals, processors, extra, send_cost, receive_cost, complex_assignment.dependent_unicast_cost_function)
    elif function == 'greedy_weak_neighbor_dependent_unicast':
        assignment_matrix = complex_assignment.greedy_with_communication(workload_matrix, width, height, intervals, processors, extra, send_cost, receive_cost, complex_assignment.weak_neighbor_dependent_unicast_cost_function)
    print(assignment_matrix)
    print_assignment(assignment_matrix, width, height)
    return assignment_matrix

def run_greedy_prioritize_communication(processors, intervals, width, height, workload_matrix, send_cost, receive_cost, tuning_constant, coarse):
    extra = -1
    assignment_matrix = complex_assignment.greedy_prioritize_communication(workload_matrix, width, height, intervals, processors, extra, send_cost, receive_cost, tuning_constant)
    print(assignment_matrix)
    print_assignment(assignment_matrix, width, height)
    return assignment_matrix

def get_dimensions(lst):
    dimensions = []
    while isinstance(lst, list):
        dimensions.append(len(lst))
        lst = lst[0] if lst else None
    return dimensions

def get_theoretical_lower_bound(width, height, input_matrix, denominator):
    value = 0
    column_max_values = [max(col) for col in zip(*input_matrix)]
    column_average_values = np.mean(input_matrix, axis=0)
    column_average_values = [x / denominator for x in column_average_values]
    for t in range(width):
        value += max(column_max_values[t], column_average_values[t])
    return value

def main():
    if len(sys.argv) < 4:
        print('Usage: python3 test_naive_greedy.py [processors] [which_greedy] [tuning_constant] [coarse]')
        return -1
    # initialization to default value
    processors = int(sys.argv[1])
    print("processors: ",processors)
    greedy_type = int(sys.argv[2])
    print("greedy_type: ", greedy_type)
    tuning_constant = float(sys.argv[3])
    print("tunning_constant: ",tuning_constant)
    coarse = int(sys.argv[4])
    print("coarse: ", coarse)
    width = 1
    height = 1
    intervals = 1
    send_cost = 5
    receive_cost = 5
    file_path = 'random_workload.csv'
    # update width
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        width = sum(1 for row in csv_reader)
    samples = width * height
    # read into the workload matrix and update interval
    workload_matrix = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=' ')
        for row in csv_reader:
            if len(row) > 0:
                row = [int(element) for element in row]
                intervals = len(row)
                workload_matrix.append(row)
    # choose the type of greedy
    assignment_matrix = None 
    if greedy_type == 2: 
        assignment_matrix = run_naive_greedy(processors, intervals, width, height, workload_matrix, send_cost, receive_cost)
    elif greedy_type == 3: 
        assignment_matrix = run_greedy_prioritize_communication(processors, intervals, width, height, workload_matrix, send_cost, receive_cost, tuning_constant, coarse)
    elif greedy_type == 4: 
        # run greedy_with_communication 
        function_list = ['greedy_dependent_unicast', 'greedy_independent_unicast', 'greedy_dependent_broadcast', 'greedy_independent_broadcast']
        function = function_list[1] 
        assignment_matrix = run_greedy_with_communication(processors, intervals, width, height, workload_matrix, send_cost, receive_cost, function)
    # get computation time across intervals 
    max_compute_time = get_computation_time_interval(workload_matrix, assignment_matrix, width, height, intervals, processors)
    # simulate computation and communication 
    totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast = simulation.simulate(workload_matrix, assignment_matrix, width, height, intervals, processors, send_cost*2, receive_cost*2)
    # calculate theoretical lower bound given the workload 
    theoretical_lower_bound = get_theoretical_lower_bound(intervals, samples, workload_matrix, processors)
    print("Final total Computation: ",totalComputation)
    print("Final total Broadcast: ",totalBroadcast)
    print("Final total Unicast: ",totalUnicast)
    print("Final total Computation Broadcast: ",totalComputationBroadcast)
    print("Final total Computation Unicast: ",totalComputationUnicast)
    print("max computation time of each interval: ",max_compute_time)
    print("theoretical_lower_bound: ",theoretical_lower_bound)
    print("efficiency ratio of computation: ", "{:.3f}".format(totalComputation/theoretical_lower_bound))
    print("efficiency ratio of computation and communication: ", "{:.3f}".format(totalComputationBroadcast/theoretical_lower_bound))

if __name__ == '__main__':
    main()