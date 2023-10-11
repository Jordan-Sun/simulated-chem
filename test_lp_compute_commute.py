import lp_computation_commute_two
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

def print_solution(solution, width, height):
    for i in range(height):
        for j in range(width):
            for element in solution[j + width * i]:
                print("{:.3f}".format(element), end=' ')
            print("\n")
    return 0

def randomly_generate_matrix(samples, processors):
    initial_assignment_matrix = np.zeros((samples,processors))
    for i in range(samples):
        initial_assignment_matrix[i][np.random.randint(0, processors)] =  1
    return initial_assignment_matrix

def dominate_generate_matrix(samples, processors):
    initial_assignment_matrix = np.zeros((samples,processors))
    p = np.random.randint(0, processors)
    for i in range(samples):
        initial_assignment_matrix[i][p] =  1
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
    final_totalComputation = float('inf')
    final_totalBroadcast = float('inf')
    final_totalUnicast = float('inf')
    final_totalComputationBroadcast = float('inf')
    final_totalComputationUnicast = float('inf')
    for iteration in range (total_iteration):
        solution = lp_computation_commute_two.lp_compute_commute(input_assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors)
        # print_solution(solution, width, height)
        assignment_matrix = np.array(lp_computation_commute_two.lp_compute_commute_max(solution, samples, processors))
        totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast = simulation.simulate(np.array(workload_matrix).reshape(samples,intervals).tolist(), assignment_matrix.tolist(), width, height, intervals, processors, send_cost, receive_cost)
        if (PrevioustotalComputationBroadcast > totalComputationBroadcast or iteration == total_iteration - 1):
            final_assignment = assignment_matrix
            PrevioustotalComputationBroadcast = totalComputationBroadcast
            print_solution(solution, width, height)
            print_matrix(assignment_matrix, width, height)

            final_totalComputation = totalComputation
            final_totalBroadcast = totalBroadcast
            final_totalUnicast = totalUnicast
            final_totalComputationBroadcast = totalComputationBroadcast
            final_totalComputationUnicast = totalComputationUnicast
        else:
            # print(iteration+1, "th round:")
            pass
            # print("This update is worse than previous ones\n")
            # formatting the output assignment
        tmp_matrix = np.zeros((samples,processors))
        for sample in range(samples):
            tmp_matrix[sample][assignment_matrix[sample]] = 1
        input_assignment_matrix = tmp_matrix
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    return final_totalComputation, final_totalBroadcast, final_totalUnicast, final_totalComputationBroadcast, final_totalComputationUnicast

def main():
    send_cost = 5
    total_iteration = 1
    receive_cost = 5
    width = 8
    height = 8
    intervals = 1
    processors = 10
    samples = width * height
    # generate workload matrix
    workload_matrix = [i for i in range(height * width * intervals)]
    # random initialization
    initial_assignment_matrix = randomly_generate_matrix(samples, processors)
    # print_solution(initial_assignment_matrix, width, height)
    # greedy initialization
    # initial_assignment_matrix = greedily_generate_matrix(np.array(workload_matrix).reshape(-1,intervals).tolist(), width, height, intervals, processors, -1)
    # dominated assignment (all tasks on one processors)
    # initial_assignment_matrix = dominate_generate_matrix(samples, processors)
    matrix = matrix_to_list(initial_assignment_matrix, samples)
    print("initial assignment: ",initial_assignment_matrix)
    totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast = simulation.simulate(np.array(workload_matrix).reshape(samples,intervals).tolist(), matrix, width, height, intervals, processors, send_cost, receive_cost)
    # print("Initial Round:")
    # print("Initial total Computation: ",totalComputation)
    # print("Initial total Broadcast: ",totalBroadcast)
    # print("Initial total Unicast: ",totalUnicast)
    # print("Initial total Computation Broadcast: ",totalComputationBroadcast)
    # print("Initial total Computation Unicast: ",totalComputationUnicast)
    # print("\n")
    final_totalComputation, final_totalBroadcast, final_totalUnicast, final_totalComputationBroadcast, final_totalComputationUnicast = test_function_lp_compute_commute(totalComputationBroadcast, workload_matrix, initial_assignment_matrix, send_cost, receive_cost, total_iteration, width, height, intervals, processors, samples)
    print("Initial total Computation: ", totalComputation, " Final total Computation: ",final_totalComputation)
    print("Initial total Broadcast: ", totalBroadcast, " Final total Broadcast: ",final_totalBroadcast)
    print("Initial total Unicast: ", totalUnicast, " Final total Unicast: ",final_totalUnicast)
    print("Initial total Computation Broadcast: ", totalComputationBroadcast, " Final total Computation Broadcast: ",final_totalComputationBroadcast)
    print("Initial total Computation Unicast: ", totalComputationUnicast, " Final total Computation Unicast: ",final_totalComputationUnicast)
    # print("workload Matrix: ",np.array(workload_matrix).reshape(samples,intervals).tolist())
    # print("assignment: ",assignment.tolist())

    return 0

if __name__ == '__main__':
    main()