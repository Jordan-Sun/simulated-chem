
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


# If ran as main, test the assignment class
if __name__ == '__main__':
    # # Test reading from nc4 file
    # assignment = Assignment.read_nc4("test/kpp_diags/GEOSChem.KppDiags.20190701_0000z.nc4")
    # print(assignment.assignment)
    # print(assignment.processors)
    # Test reading from csv file
    assignment = Assignment.read_csv("test/og_assignments/c24_p24.csv")
    print(assignment.assignment)
    print(assignment.processors)
    # Test writing to csv file
    df = pd.DataFrame([i for i in range(6) for _ in range(576)], columns=['KppRank'])
    assignment = Assignment(df)
    assignment.write_csv("test/og_assignments/c24_p6.csv")
    # Test reading back from csv file
    assignment = Assignment.read_csv("test/og_assignments/c24_p6.csv")
    print(assignment.assignment)
    print(assignment.processors)
