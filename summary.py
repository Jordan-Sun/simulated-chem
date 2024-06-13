# summaries how many columns are assigned to each processor for a given assignment.
import sys
import pandas as pd

# check if the number of arguments is correct
if len(sys.argv) != 2:
    print("Usage: python summary.py <path>")
    sys.exit(1)
# get the assignment file from the first argument
file = sys.argv[1]
# create a dictionary to store the number of columns assigned to each processor
count = {}
# read the assignment file as a csv file
df = pd.read_csv(file, header=None)
# count the number of columns assigned to each processor
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if df.iloc[i, j] not in count:
            count[df.iloc[i, j]] = 0
        count[df.iloc[i, j]] += 1
# print the count
print(count)
# print the maximum number of columns assigned to a processor
print("Max: ", max(count.values()))
# print the minimum number of columns assigned to a processor
print("Min: ", min(count.values()))
# print the total number of columns
print("Total: ", sum(count.values()))