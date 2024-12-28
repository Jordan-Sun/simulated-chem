from workload import Workload

import os
from dataclasses import dataclass, field
from typing import Tuple

import netCDF4 as nc
import numpy as np
import pandas as pd

# Defines the assignment class, a dataclass to store the assignment matrix
@dataclass
class Assignment:
    assignment: pd.DataFrame = field(default=None)
    samples: int = field(init=False, default=0)
    intervals: int = field(init=False, default=0)
    processors: int = field(init=False, default=0)

    def __post_init__(self):
        if self.assignment is not None:
            # The number of samples is the number of rows in the assignment matrix
            self.samples, self.intervals = self.assignment.shape
            # The number of processors is the maximum value in the assignment matrix plus one
            self.processors = np.max(self.assignment.values) + 1

    # Concatenates a list of assignments into a single assignment
    @staticmethod
    def concatenate(assigns: list['Assignment']) -> 'Assignment':
        # Concatenate the assignments
        assignments = [assign.assignment for assign in assigns]
        return Assignment(pd.concat(assignments, axis=1))

    # Reads the assignment from a NC4 file
    @staticmethod
    def read_nc4(file_name: os.path) -> 'Assignment':
        # Open the netcdf file
        with nc.Dataset(file_name, 'r') as f:
            # Read the KppRank variable
            assignment = f.variables['KppRank'][:]
            # Convert the masked array to a ndarray and flatten it
            assignment = assignment.filled().flatten()
        return Assignment(pd.DataFrame(assignment))

    # Reads the assignment from a csv file
    @staticmethod
    def read_csv(file_name: os.path) -> 'Assignment':
        # Read the assignment from the file
        assignment = pd.read_csv(file_name, index_col=0)
        return Assignment(assignment)
    
    # Writes the assignment to a csv file
    def write_csv(self, file_name: os.path):
        self.assignment.to_csv(file_name)

    # Writes the assignment to a directory of mapping files for each processor
    def write_mapping(self, original_assignment: 'Assignment', directory: os.path):
        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Matrix for each processor to store the mapping
        # The first space is used to store which processor should the processor expect to receive samples
        sources = [[-1 for _ in range(self.intervals)] for _ in range(self.processors)]
        targets = [[-1 for _ in range(self.intervals)] for _ in range(self.processors)]
        mapping = [[[] for _ in range(
            self.intervals)] for _ in range(self.processors)]
        # Iterate over the intervals
        for interval in range(self.intervals):
            # Reset the index counter for each interval
            index_counter = [0 for _ in range(self.processors)]
            # Iterate over the samples
            for sample in range(self.samples):
                # Obtain the processor from the original assignment
                source = original_assignment.assignment.iloc[sample, 0]
                # Increment the index counter for the source
                index_counter[source] += 1
                # Obtain the processor to which the sample is assigned
                target = self.assignment.iloc[sample, interval]
                # Set the source's target and the target's source and add the sample to the mapping
                if source != target:
                    sources[target][interval] = source
                    targets[source][interval] = target
                    mapping[source][interval].append(index_counter[source])
        # Write the mapping to the directory
        for processor in range(self.processors):
            with open(os.path.join(directory, f"rank_{processor}.csv"), 'w') as f:
                # Compute the maximum length (the longest line should always be one of the mapping array lines)
                max_length = max([len(mapping[processor][interval]) for interval in range(self.intervals)])
                # Print the number of intervals and maximum length
                f.write(f"{self.intervals}, {max_length}\n")
                # Print the mapping for each interval in one line
                for interval in range(self.intervals):
                    # Source rank # to recv from
                    f.write(f"{sources[processor][interval]},")
                    # Target rank  # to send to
                    f.write(f"{targets[processor][interval]},")
                    # Length of the array to send
                    f.write(f"{len(mapping[processor][interval])}\n")
                    # Array of columns to send on a new line for Fortran to process
                    f.write(",".join([str(rank) for rank in mapping[processor][interval]]))
                    f.write("\n")

    # Simulates the assignment for a given workload
    def simulate(self, workload: 'Workload', static: bool = False, sim_log: str = None) -> float:
        # Initialize the simulated workload 
        L_sim = 0
        # Open the log file if it is provided
        if sim_log is not None:
            f = open(sim_log, 'w')
            f.write("Interval," + ",".join([f"Processor{i}" for i in range(self.processors)]) + ",Max\n")
        # Iterate over the intervals of the workload if static
        if static:
            intervals = range(workload.intervals)
        # Iterate over the intervals of the assignment if not static
        else:
            intervals = range(self.intervals)

        for interval in intervals:
            # Store the workload for each processor in a list
            L_int = [0 for _ in range(self.processors)]
            # Iterate over the samples
            for sample in range(self.samples):
                # Obtain the processor to which the sample is assigned
                if static:
                    processor = self.assignment.iloc[sample, 0]
                else:
                    processor = self.assignment.iloc[sample, interval]
                # Add the workload to the processor
                L_int[processor] += workload.workload.iloc[sample, interval]
            # Add the maximum workload to the simulated workload
            L_sim += max(L_int)    
            # Write the interval workload to the log file
            if sim_log is not None:
                f.write(f"{interval}," + ",".join([str(L) for L in L_int]) + f",{max(L_int)}\n")
        # Write the total workload to the log file
        if sim_log is not None:
            f.write(f"Total,{L_sim}\n")
        return L_sim
    
    # Movement of samples between processors
    def movement(self, original_assignment: 'Assignment', send_log: str = None, recv_log: str = None) -> Tuple[int, int, int, int]:
        # Initialize the samples sent and received
        S_sum = 0
        R_sum = 0
        S_max = 0
        R_max = 0
        # Open the log file if it is provided
        if send_log is not None:
            f = open(send_log, 'w')
            f.write("Interval," + ",".join([f"Processor{i}" for i in range(self.processors)]) + ",Total,Max\n")
        if recv_log is not None:
            g = open(recv_log, 'w')
            g.write("Interval," + ",".join([f"Processor{i}" for i in range(self.processors)]) + ",Total,Max\n")
        # Iterate over the intervals
        for interval in range(self.intervals):
            # Store the samples sent and received for each processor in a list
            S_int = [0 for _ in range(self.processors)]
            R_int = [0 for _ in range(self.processors)]
            # Store the send recv pairs for each processor
            pairs = set()
            # Iterate over the samples
            for sample in range(self.samples):
                # Obtain the processor from the original assignment
                source = original_assignment.assignment.iloc[sample, 0]
                # Obtain the processor to which the sample is assigned
                target = self.assignment.iloc[sample, interval]
                # Increment the samples sent and received
                if source != target:
                    # Add the pair to the set and increment the samples sent and received
                    pairs.add((source, target))
                    S_int[source] += 1
                    R_int[target] += 1
            # Verify that each processor is in one send recv pair as source and one as target
            sources = [0 for _ in range(self.processors)]
            targets = [0 for _ in range(self.processors)]
            for source, target in pairs:
                sources[source] += 1
                targets[target] += 1
            for i in range(self.processors):
                if sources[i] != 1:
                    print(f"Error: Processor {i} is not in exactly one send pair at interval {interval}")
                    print(f"Pairs: {pairs}")
                if targets[i] != 1:
                    print(f"Error: Processor {i} is not in exactly one recv pair at interval {interval}")
                    print(f"Pairs: {pairs}")
            # Verify that the samples sent is equal to the samples received for each pair
            for source, target in pairs:
                if S_int[source] != R_int[target]:
                    print(f"Error: Processor {source} sent {S_int[source]} samples but processor {target} received {R_int[target]} samples at interval {interval}")
                    print(f"Pairs: {pairs}")
            # Add the samples sent and received to the total and max
            S_sum += sum(S_int)
            R_sum += sum(R_int)
            S_max += max(S_int)
            R_max += max(R_int)
            # Write the interval samples sent and received to the log files
            if send_log is not None:
                f.write(f"{interval}," + ",".join([str(S) for S in S_int]) + f",{sum(S_int)},{max(S_int)}\n")
            if recv_log is not None:
                g.write(f"{interval}," + ",".join([str(R) for R in R_int]) + f",{sum(R_int)},{max(R_int)}\n")
        # Write the total and max samples sent and received to the log files
        if send_log is not None:
            f.write(f"Total,{S_sum},Max,{S_max}\n")
        if recv_log is not None:
            g.write(f"Total,{R_sum},Max,{R_max}\n")
        return S_sum, R_sum, S_max, R_max


# If ran as main, test the assignment class
if __name__ == '__main__':
    # Read the workload
    workload = Workload.read_csv("test/workloads/c24.csv")

    # # Test reading from nc4 file
    # assignment = Assignment.read_nc4("test/kpp_diags/GEOSChem.KppDiags.20190701_0000z.nc4")
    # print(assignment.assignment)
    # print(assignment.processors)

    # print("Testing c24 p24 original")
    # # Test reading from csv file
    # print("Test reading from csv file")
    # og_assignment = Assignment.read_csv(
    #     "test/og_assignments/c24_p24.csv")
    # assert og_assignment.assignment.shape == (3456, 1)
    # assert og_assignment.processors == 24
    # # Test simulate
    # print("Test simulate")
    # L = og_assignment.simulate(workload, True, "test/og_assignments/c24_p24_simulation.csv")
    # print(L)

    print("Testing c24 p6 original")
    # Test reading back from csv file
    print("Test reading from csv file")
    og_assignment = Assignment.read_csv("test/og_assignments/c24_p6.csv")
    assert og_assignment.assignment.shape == (3456, 1)
    assert og_assignment.processors == 6
    # # Test simulate
    # print("Test simulate")
    # L = og_assignment.simulate(
    #     workload, True, "test/og_assignments/c24_p6_simulation.csv")
    # print(L)
    
    print("Testing c24 p6 MIQCP")
    test_path = "test/MIQCP/c24_p6_b1.0"
    # Test read MIQCP assignment
    print("Test reading from csv file")
    assignment = Assignment.read_csv(f"{test_path}/assignment.csv")
    assert assignment.assignment.shape == (3456, 72)
    assert assignment.processors == 6
    # Test write mapping
    print("Test write mapping")
    assignment.write_mapping(og_assignment, f"{test_path}/mappings")
    # Test simulate
    print("Test simulate")
    L = assignment.simulate(workload, False, f"{test_path}/simulation.csv")
    print(L)
    # Test movement
    print("Test movement")
    S_sum, R_sum, S_max, R_max = assignment.movement(og_assignment, f"{test_path}/send.csv", f"{test_path}/recv.csv")
    print(S_sum, R_sum, S_max, R_max)
