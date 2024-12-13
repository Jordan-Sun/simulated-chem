import os
import netCDF4 as nc
import numpy as np
import pandas as pd

# Defines the workload class
class Workload:
    # Constructor
    def __init__(self, workload: pd.DataFrame = None):
        # Create a dataframe from the costs
        self.workload = workload
        # If the workload is not None
        if self.workload is not None:
            # The number of samples is the number of rows in the workload matrix
            # The number of intervals is the number of columns in the workload matrix
            self.samples, self.intervals = self.workload.shape

    # Reads raw workload from raw nc4 format to a numpy array
    @staticmethod
    def read_nc4(file_name: os.path) -> np.ndarray:
        # Open the netcdf file
        with nc.Dataset(file_name, 'r') as f:
            # Read only the KppTotSteps variable
            var = f.variables['KppTotSteps'][:]
            # Convert the masked array to a ndarray
            var = var.filled()
            # Round up to integer
            var = np.ceil(var).astype(int)
        # Return the costs
        costs = var[0].sum(axis=0).flatten()
        return costs
    
    # Reads raw workload from a nc4 file
    @staticmethod
    def read_nc4_file(file_name: os.path) -> 'Workload':
        # Read the workload from the file
        workload = Workload.read_nc4(file_name)
        return Workload(pd.DataFrame(workload))
    
    # Reads raw workload from a directory of nc4 files
    @staticmethod
    def read_nc4_dir(dir_name: os.path) -> 'Workload':
        # Dictionary to store the workloads by timestamp
        workloads = {}
        for filename in os.listdir(dir_name):
            if filename.startswith('GEOSChem.KppDiags.') and filename.endswith('.nc4'):
                # Convert the file name to timestamp
                parts = os.path.basename(filename).split('.')
                # Extract the timestamp from the parts
                timestamp = parts[2]
                # Read the workload from the file
                workload = Workload.read_nc4_file(os.path.join(dir_name, filename))
                # Store the workload in the dictionary
                workloads[timestamp] = workload.workload
        # Sort the workloads by timestamp
        workloads = dict(sorted(workloads.items()))
        # Concatenate the workloads into a single dataframe with the timestamps as column names
        df = pd.concat(workloads.values(), axis=1)
        df.columns = workloads.keys()
        return Workload(df)

    # Reads workload from processed csv format
    @staticmethod
    def read_csv(file_name: os.path) -> 'Workload':
        # Read the workload from the file
        workload = pd.read_csv(file_name, index_col=0)
        # Create a Workload object from the dataframe
        return Workload(workload)
    
    # Writes the workload to a csv file
    def write_csv(self, file_name: os.path):
        self.workload.to_csv(file_name)

# If ran as main, test the workload class
if __name__ == "__main__":
    # Test reading from nc4 file
    workload = Workload.read_nc4_file("test/kpp_diags/GEOSChem.KppDiags.20190701_0000z.nc4")
    print(workload.workload)
    # Test reading from nc4 directory
    workload = Workload.read_nc4_dir("test/kpp_diags")  
    print(workload.workload)
    # Test reading from csv file
    workload = Workload.read_csv("test/workload.csv")
    print(workload.workload)
