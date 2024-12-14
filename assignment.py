from workload import Workload

import os
from dataclasses import dataclass, field
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
        mapping = [[[] for _ in range(self.intervals)] for _ in range(self.processors)]
        # Iterate over the intervals
        for interval in range(self.intervals):
            # Iterate over the samples
            for sample in range(self.samples):
                # Obtain the processor from the original assignment
                source = original_assignment.assignment['KppRank'][sample]
                # Obtain the processor to which the sample is assigned
                target = self.assignment['KppRank'][sample]
                # Add the sample to the processor's mapping
                mapping[source][interval].append(target)
        # Write the mapping to the directory
        for processor in range(self.processors):
            with open(os.path.join(directory, f"rank_{processor}.txt"), 'w') as f:
                df = pd.DataFrame(mapping[processor])
                df.to_csv(f, header=False, index=False)

    # Simulates the assignment for a given workload
    def simulate(self, workload: 'Workload', sim_log: str = None) -> int:
        # Initialize the simulated workload 
        L_sim = 0
        # Open the log file if it is provided
        if sim_log is not None:
            f = open(sim_log, 'w')
            f.write("Interval," + ",".join([f"Processor{i}" for i in range(self.processors)]) + ",Max\n")
        # Iterate over the intervals
        for interval in range(self.intervals):
            # Store the workload for each processor in a list
            L_int = [0 for _ in range(self.processors)]
            # Iterate over the samples
            for sample in range(self.samples):
                # Obtain the processor to which the sample is assigned
                processor = self.assignment.iloc[sample, interval]
                # Add the workload to the processor
                L_int[processor] += workload.workload.iloc[sample, interval]
            # Add the maximum workload to the simulated workload
            L_sim += max(L_int)    
            # Write the interval workload to the log file
            if sim_log is not None:
                f.write(f"{interval}," + ",".join([str(L) for L in L_int]) + f",{max(L_int)}\n")
        return L_sim


# If ran as main, test the assignment class
if __name__ == '__main__':
    # # Test reading from nc4 file
    # assignment = Assignment.read_nc4("test/kpp_diags/GEOSChem.KppDiags.20190701_0000z.nc4")
    # print(assignment.assignment)
    # print(assignment.processors)
    # # Test reading from csv file
    # og_assignment = Assignment.read_csv(
    #     "test/og_assignments/c24_p24.csv")
    # print(og_assignment.assignment)
    # print(og_assignment.processors)
    # # Test writing to csv file
    # df = pd.DataFrame([i for i in range(6) for _ in range(576)], columns=['KppRank'])
    # og_assignment = Assignment(df)
    # og_assignment.write_csv("test/og_assignments/c24_p6.csv")
    # Test reading back from csv file
    og_assignment = Assignment.read_csv("test/og_assignments/c24_p6.csv")
    print(og_assignment.assignment)
    print(og_assignment.processors)
    # Test read MIQCP assignment
    assignment = Assignment.read_csv("test/MIQCP/c24_p6/assignment.csv")
    print(assignment.assignment)
    print(assignment.processors)
    # Test write mapping
    assignment.write_mapping(og_assignment, "test/MIQCP/c24_p6/mappings")
    # Test simulate
    workload = Workload.read_csv("test/workloads/c24.csv")
    L_sim = assignment.simulate(workload, "test/MIQCP/c24_p6/simulation.csv")
