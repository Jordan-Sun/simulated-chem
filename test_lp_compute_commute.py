import lp_computation_commute
import numpy as np

def print_matrix(matrix, width, height):
    for i in range(height):
        for j in range(width):
            print(matrix[j + width * i], end=' ')
        print("\n")
    return 0

def main():
    send_cost = 1
    receive_cost = 1
    width = 3
    height = 3
    intervals = 1
    processors = 3
    samples = width * height
    workload_matrix = [2 * i + 2 for i in range(height * width * intervals)]
    # print_matrix(workload_matrix, intervals, samples)
    # print("samples: ",samples)
    # print("intervals: ",intervals)
    # print("processors: ",processors)
    # print("workload_matrix shape: ",workload_matrix.shape)
    # print("workload_matrix: ",workload_matrix)
    solution = lp_computation_commute.lp_compute_commute(send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors)
    print("solution",solution)
    solution = lp_computation_commute.lp_compute_commute_max(solution, samples, processors)
    assignment = np.array(solution)
    print_matrix(assignment, width, height)
    return 0

if __name__ == '__main__':
    main()