import os
import numpy as np
import pandas as pd

# from scipy.ndimage import zoom
from skimage.transform import resize

# Defines the workload class
class Workload:
    # Constructor
    def __init__(self, workload: pd.DataFrame = None):
        # Create a dataframe from the costs
        self.workload = workload
        # Workload attributes
        self.intervals = 0
        self.samples = 0
        self.resolution = 0
        # If the workload is not None
        if self.workload is not None:
            # The number of samples is the number of rows in the workload matrix
            # The number of intervals is the number of columns in the workload matrix
            self.samples, self.intervals = self.workload.shape
            # The resolution is the square root of the number of samples divided by 6
            self.resolution = np.sqrt(self.samples / 6)
            # Make sure the resolution is an integer, else throw a warning
            if self.resolution != int(self.resolution):
                raise ValueError(f"Error: resolution {self.resolution} is not an integer")
            self.resolution = int(self.resolution)

    # Reads raw workload from raw nc4 format to a numpy array
    @staticmethod
    def read_nc4(file_name: os.path) -> np.ndarray:
        import netCDF4 as nc  # Import netCDF4 only when needed
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

    # Computes the lower bound of the workload for a given number of processors for the given intervals if given
    def lower_bound(self, processors: int, intervals: list[int] = None) -> int:
        # If intervals is not given, use all intervals
        if intervals is None:
            intervals = range(self.intervals)
        # The bound is the sum of span at each interval
        L_bound = 0
        for interval in intervals:
            # The span of the interval is the sum of the workload at the interval divided by the number of processors
            w_max = self.workload.iloc[:, interval].max()
            L_avg = self.workload.iloc[:, interval].sum() / processors
            # Add the maximum of the average and maximum workload to the bound, also ceil it because fraction is not possible
            L_bound += np.ceil(max(L_avg, w_max))
        return int(L_bound)

    # Upscales the workload to a different resolution
    def upscale(self, target_resolution: int, order: int = 0) -> 'Workload':
        # Calculate the scale factor
        scale_factor = target_resolution / self.resolution
        # Reshape the workload matrix to 6 * res * res by intervals
        reshaped_workload = self.workload.values.reshape(6, self.resolution, self.resolution, self.intervals)

        # # Apply zoom for upscaling (bad because it uses edge interpolation)
        # upscaled_workload = zoom(reshaped_workload, (1.0, scale_factor, scale_factor, 1.0), order=order)

        # Use skimage's resize for bin interpretation
        upscaled_workload = resize(
            reshaped_workload,
            (6, target_resolution, target_resolution, self.intervals),
            order=order,
            preserve_range=True,  # Prevents normalization
            anti_aliasing=False,  # For area/binned interpretation
        )

        # Reshape back to 6 * target_res * target_res by intervals
        upscaled_workload = upscaled_workload.reshape(6 * target_resolution * target_resolution, self.intervals)
        # Create a new Workload object from the upscaled workload, preserving column headers
        return Workload(pd.DataFrame(upscaled_workload, columns=self.workload.columns))

# If ran as main, test the workload class
if __name__ == "__main__":
    # Read the workload
    workload = Workload.read_csv("test/workloads/c24.csv")
    # Upscale the workload to a different resolution
    target_resolution = 90
    upscale_dict = {
        # 0: "nearest",
        1: "bilinear",
        # 3: "bicubic",
    }
    for order, method in upscale_dict.items():
        # Upscale the workload
        upscaled_workload = workload.upscale(target_resolution, order=order)
        # Write the workload to a csv file
        upscaled_workload.write_csv(
            f"test/workloads/{method}_c24_to_c{target_resolution}.csv"
        )

    # # Read the actual workload from the file
    # workload = Workload.read_csv(f"test/workloads/c{target_resolution}.csv")

    # # Test computing lower bound
    # intervals = range(72)
    # print(workload.lower_bound(36, intervals))
    # print(workload.lower_bound(144, intervals))
    # print(workload.lower_bound(576, intervals))
