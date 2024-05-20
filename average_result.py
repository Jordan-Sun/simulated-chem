import os
import scipy.stats as stats

def find_file(root_folder, filename):
    for root, dirs, files in os.walk(root_folder):
        if filename in files:
            return os.path.join(root, filename)
    return None

import os
import scipy.stats as stats

def main():
    tuning_constant_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    coarse_constant_list = [2,3,4,5]
    two_dim_array = [[0 for _ in range(len(tuning_constant_list))] for _ in range(len(coarse_constant_list))]
    index_list = list(range(1,500,1))
    root_path = './feather/p_54/'
    total_count = 0
    confidence_level = 0.95
    confidence_level_array = [[0 for _ in range(len(tuning_constant_list))] for _ in range(len(coarse_constant_list))]
    for row_index, coarse_constant in enumerate(coarse_constant_list):
        for col_index, tuning_constant in enumerate(tuning_constant_list):
            total_cost = []
            for index in index_list:
                formatted_file_name = f"greedy_prioritize_communication_{index}_send10_recv20_tuning{tuning_constant}_coarse{coarse_constant}.result"
                file_path = os.path.join(root_path, formatted_file_name)
                # Read the file from file path given
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        # Read lines one by one
                        for line in file:
                            if line.startswith('ComputationUnicast'):
                                # Extract and return the number
                                total_count += 1
                                tmp = float(line.split(',')[1].strip())
                                total_cost.append(tmp)
            # Calculate the average and store it in the array
            two_dim_array[row_index][col_index] = sum(total_cost) / len(total_cost)
            # Calculate CI
            sample_mean = sum(total_cost) / len(total_cost)
            sample_std = stats.tstd(total_cost)
            df = len(total_cost) - 1
            t_critical = stats.t.ppf((1 + confidence_level) / 2, df)  # t-critical value for 95% CI
            margin_of_error = t_critical * (sample_std / (len(total_cost) ** 0.5))
            confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
            confidence_level_array[row_index][col_index] = confidence_interval

    # Print each row of the table
    with open('output.txt', 'w') as f:
        # Write the header for the tuning constants
        header = " " * 7  # Initial space for the coarse constant column
        for tuning_constant in tuning_constant_list:
            header += f"{tuning_constant:>15}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")  # Divider

        # Write the averages table
        for row_index, coarse_constant in enumerate(coarse_constant_list):
            f.write(f"{coarse_constant:<7}")  # 7-character wide, left-aligned
            for col_index, tuning_constant in enumerate(tuning_constant_list):
                # Write each cell with wide spacing
                f.write(f"{two_dim_array[row_index][col_index]:>15.2f}")
            f.write("\n")  # Newline after each row

        # Write the header again for the confidence intervals
        header = " " * 15  # Increased space for the coarse constant column
        for tuning_constant in tuning_constant_list:
            header += f"{tuning_constant:>30}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")  # Divider

        for row_index, coarse_constant in enumerate(coarse_constant_list):
            f.write(f"{coarse_constant:<15}")  # Adjusted to 15-character wide for alignment
            for col_index, tuning_constant in enumerate(tuning_constant_list):
                # Format and write each confidence interval cell with wide spacing
                lower_bound, upper_bound = confidence_level_array[row_index][col_index]
                f.write(f"({lower_bound:11.2f}, {upper_bound:15.2f})")  # Removed '>' for left alignment
            f.write("\n")  # Newline after each row

    print("total count :", total_count)

if __name__ == "__main__":
    main()
