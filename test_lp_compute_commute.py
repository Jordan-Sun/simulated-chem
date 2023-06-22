import lp_computation_commute
import simulation
import numpy as np
import time

def print_matrix(matrix, width, height):
    for i in range(height):
        for j in range(width):
            print(matrix[j + width * i], end=' ')
        print("\n")
    return 0

def main():
    send_cost = 5
    total_iteration = 10
    receive_cost = 10
    width = 10
    height = 10
    intervals = 20
    processors = 5
    samples = width * height
    # generate workload matrix
    workload_matrix = [2 * i + 2 for i in range(height * width * intervals)]
    # random guess assignment matrix
    previous_assignment_matrix = np.zeros((samples,processors))
    for i in range(samples):
        previous_assignment_matrix[i][np.random.randint(0, processors)] =  1
    # print_matrix(workload_matrix, intervals, samples)
    print("samples: ",samples)
    print("intervals: ",intervals)
    print("processors: ",processors)
    # print("workload_matrix shape: ",workload_matrix.shape)
    # print("workload_matrix: ",workload_matrix)
    matrix = []
    for sample in range(samples):
        matrix.append(np.where(previous_assignment_matrix[sample] == 1)[0][0])
    totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast = simulation.simulate(np.array(workload_matrix).reshape(samples,intervals).tolist(), matrix, width, height, intervals, processors, send_cost, receive_cost)
    print("total Computation: ",totalComputation)
    print("total Broadcast: ",totalBroadcast)
    print("total Unicast: ",totalUnicast)
    print("total Computation Broadcast: ",totalComputationBroadcast)
    print("total Computation Unicast: ",totalComputationUnicast)
    start_time = time.time()
    PrevioustotalComputationBroadcast = float('inf')
    for iteration in range (total_iteration):
        solution = lp_computation_commute.lp_compute_commute(previous_assignment_matrix, send_cost, receive_cost, width, height, workload_matrix, samples, intervals, processors)
        assignment_matrix = np.array(lp_computation_commute.lp_compute_commute_max(solution, samples, processors))
        totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast = simulation.simulate(np.array(workload_matrix).reshape(samples,intervals).tolist(), assignment_matrix.tolist(), width, height, intervals, processors, send_cost, receive_cost)
        print("total Computation: ",totalComputation)
        print("total Broadcast: ",totalBroadcast)
        print("total Unicast: ",totalUnicast)
        print("total Computation Broadcast: ",totalComputationBroadcast)
        print("total Computation Unicast: ",totalComputationUnicast)
        print("\n")
        if (PrevioustotalComputationBroadcast>totalComputationBroadcast):
            previous_assignment_matrix = assignment_matrix
        # formatting the output assignment
        tmp_matrix = np.zeros((samples,processors))
        for sample in range(samples):
            tmp_matrix[sample][previous_assignment_matrix[sample]] = 1
        previous_assignment_matrix = tmp_matrix
        
    # print("solution",solution)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    assignment = lp_computation_commute.lp_compute_commute_random(solution, samples, processors)
    assignment = np.array(assignment)
    print_matrix(assignment, width, height)
    # print("workload Matrix: ",np.array(workload_matrix).reshape(samples,intervals).tolist())
    # print("assignment: ",assignment.tolist())
    # totalComputation, totalBroadcast, totalUnicast, totalComputationBroadcast, totalComputationUnicast = simulation.simulate(np.array(workload_matrix).reshape(samples,intervals).tolist(), assignment.tolist(), width, height, intervals, processors, send_cost, receive_cost)
    # print("total Computation: ",totalComputation)
    # print("total Broadcast: ",totalBroadcast)
    # print("total Unicast: ",totalUnicast)
    # print("total Computation Broadcast: ",totalComputationBroadcast)
    # print("total Computation Unicast: ",totalComputationUnicast)
    # print("Elapsed time:", elapsed_time, "seconds")

    return 0

if __name__ == '__main__':
    main()