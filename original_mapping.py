import os

# Writes the assignment to a directory of mapping files for each processor
def write_mapping(processors: int, intervals: int, directory: os.path):
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Write the mapping to the directory
    for processor in range(processors):
        with open(os.path.join(directory, f"rank_{processor}.csv"), 'w') as f:
            # Print the number of intervals and maximum length
            base_length = len(str(intervals)) + 2
            max_length = base_length
            while True:
                verify_length = base_length + len(str(max_length))
                if verify_length == max_length:
                    break
                max_length = verify_length
            max_length = max(max_length, 10)
            f.write(f"{intervals},{max_length}\n")
            # Print the mapping for each interval in one line
            for _ in range(intervals):
                f.write("-1,-1,0\n\n")

write_mapping(6, 72, "test/og_assignments/c24_p6/mappings")
