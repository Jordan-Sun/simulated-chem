import lp_computation_commute
import simulation
import numpy as np
import time
import complex_assignment
def print_matrix(matrix, width, height):
    for i in range(height):
        for j in range(width):
            print(matrix[j + width * i], end=' ')
        print("\n")
    return 0

def randomly_generate_matrix(samples, processors):
    initial_assignment_matrix = np.zeros((samples,processors))
    for i in range(samples):
        initial_assignment_matrix[i][np.random.randint(0, processors)] =  1
    return initial_assignment_matrix

def list_to_matrix(input_list, height, width):
    output_matrix = np.zeros((height, width))
    # print(output_matrix)
    for i in range(height):
        output_matrix[i][input_list[i]] = 1
    return output_matrix

def matrix_to_list(input_matrix, height):
    output_list = []
    for j in range(height):
        output_list.append(np.where(input_matrix[j] == 1)[0][0])
    return output_list

def greedily_generate_matrix(workload_matrix, width, height, intervals, processors, extra):
    initial_assignment = complex_assignment.greedy(workload_matrix, width, height, intervals, processors, extra)
    # print(initial_assignment)
    return list_to_matrix(initial_assignment, width * height, processors)

def test_function_lp_compute_commute(input_totalComputationBroadcast, workload_matrix, input_assignment_matrix, send_cost, receive_cost, total_iteration, width, height, intervals, processors, samples):
    PrevioustotalComputationBroadcast = input_totalComputationBroadcast
    start_time = time.time()
    final_assignment = input_assignment_matrix
    for iteration in range (total_iteration):
        solution = lp_computation_commute.lp_compute_commute(input_assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors)
        assignment_matrix = np.array(lp_computation_commute.lp_compute_commute_max(solution, samples, processors))
        totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast = simulation.simulate(np.array(workload_matrix).reshape(samples,intervals).tolist(), assignment_matrix.tolist(), width, height, intervals, processors, send_cost, receive_cost)
        if (PrevioustotalComputationBroadcast > totalComputationBroadcast):
            final_assignment = assignment_matrix
            PrevioustotalComputationBroadcast = totalComputationBroadcast
            print(iteration+1, "th round:")
            print("total Computation: ",totalComputation)
            print("total Broadcast: ",totalBroadcast)
            print("total Unicast: ",totalUnicast)
            print("total Computation Broadcast: ",totalComputationBroadcast)
            print("total Computation Unicast: ",totalComputationUnicast)
            print("\n")
        else:
            print(iteration+1, "th round:")
            print("This update is worse than previous ones\n")
        # formatting the output assignment
        tmp_matrix = np.zeros((samples,processors))
        for sample in range(samples):
            tmp_matrix[sample][assignment_matrix[sample]] = 1
        input_assignment_matrix = tmp_matrix
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

def main():
    send_cost = 5
    total_iteration = 10
    receive_cost = 5
    width = 10
    height = 10
    intervals = 5
    processors = 3
    samples = width * height
    # generate workload matrix
    workload_matrix = [2 * i + 2 for i in range(height * width * intervals)]
    # random initialization
    # initial_assignment_matrix = randomly_generate_matrix(samples, processors)
    # greedy initialization
    initial_assignment_matrix = greedily_generate_matrix(np.array(workload_matrix).reshape(-1,intervals).tolist(), width, height, intervals, processors, -1)
    matrix = matrix_to_list(initial_assignment_matrix, samples)
    
    totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast = simulation.simulate(np.array(workload_matrix).reshape(samples,intervals).tolist(), matrix, width, height, intervals, processors, send_cost, receive_cost)
    print("Initial Round:")
    print("total Computation: ",totalComputation)
    print("total Broadcast: ",totalBroadcast)
    print("total Unicast: ",totalUnicast)
    print("total Computation Broadcast: ",totalComputationBroadcast)
    print("total Computation Unicast: ",totalComputationUnicast)
    print("\n")
    test_function_lp_compute_commute(totalComputationBroadcast, workload_matrix, initial_assignment_matrix, send_cost, receive_cost, total_iteration, width, height, intervals, processors, samples)
    # print("workload Matrix: ",np.array(workload_matrix).reshape(samples,intervals).tolist())
    # print("assignment: ",assignment.tolist())

    return 0

if __name__ == '__main__':
    main()