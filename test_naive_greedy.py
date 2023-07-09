import simulation
import numpy as np
import time
import csv
import complex_assignment

def simulate(workload_matrix, assignment_matrix, width, height, intervals, processors):
    # Validate the input
    samples = width * height
    if samples != len(workload_matrix):
        raise ValueError('Invalid number of samples, expected ' + str(len(workload_matrix)) + ', got ' + str(samples) + '.')
    if intervals != len(workload_matrix[0]):
        raise ValueError('Invalid number of intervals, expected ' + str(len(workload_matrix[0])) + ', got ' + str(intervals) + '.')
    totalComputation = 0
    print(workload_matrix)
    print(assignment_matrix)
    # convert assignment matrix into another form
    helper_assignment_matrix = np.zeros((samples, processors))
    for sample in range(samples):
        helper_assignment_matrix[sample][assignment_matrix[sample]] = 1
    compute_time_matrix = helper_assignment_matrix.T @ np.array(workload_matrix)
    max_compute_time = [max(column) for column in zip(*compute_time_matrix)]
    return max_compute_time, sum(max_compute_time)

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

def run_naive_greedy(processors, intervals, width, height, workload_matrix):
    extra = -1
    sendCost = 0
    recvCost = 0
    assignment_matrix = complex_assignment.greedy(workload_matrix, width, height, intervals, processors, extra)
    print(assignment_matrix)
    print_assignment(assignment_matrix, width, height)
    max_compute_time, totalComputation = simulate(workload_matrix, assignment_matrix, width, height, intervals, processors)
    return max_compute_time, totalComputation

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
    #initialization
    width = 1
    height = 1
    intervals = 5
    processors = 3
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
            row = [int(element) for element in row]
            intervals = len(row)
            workload_matrix.append(row)
    # get cost of greedy algorithm
    max_compute_time, greedy_cost = run_naive_greedy(processors, intervals, width, height, workload_matrix)
    theoretical_lower_bound = get_theoretical_lower_bound(intervals, samples, workload_matrix, processors)
    print("max time of each interval: ",max_compute_time)
    print("total cost of greedy: ",greedy_cost)
    print("theoretical_lower_bound: ",theoretical_lower_bound)
    print("efficiency ratio: ", "{:.3f}".format(greedy_cost/theoretical_lower_bound))

if __name__ == '__main__':
    main()