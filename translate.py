import pandas as pd
import pyarrow.feather as feather

# initialize dataframe
length = 432
tasks = 48600
workload_df = pd.DataFrame(index=range(tasks), columns=range(length))

# populate the dataframe
for step in range(length):
    with open(f"chemcost_step{step:05d}.feather", "rb") as f:
        step_df = feather.read_feather(f)
        workload_df[step] = step_df["cost"]

# write the dataframe to a workloads file
with open("chemcost.workload", "wb") as f:
    workload_df.to_csv(f)