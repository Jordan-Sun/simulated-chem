import pandas as pd

def test_processor_steps_pandas():
    sim_df = pd.read_csv("test/MIQCP_same/c24_p6_b1.0/simulation.csv")
    send_df = pd.read_csv("test/MIQCP_same/c24_p6_b1.0/send.csv")
    for i in range(len(sim_df)):
        sim_max = sim_df.iloc[i, -1]
        for p in range(1, 7):
            if send_df.iloc[i, p] == 0:
                if sim_df.iloc[i, p] >= sim_max:
                    raise ValueError(f"Interval {sim_df.iloc[i, 0]}: Processor {p-1} fails check.")
                else:
                    print(f"Interval {sim_df.iloc[i, 0]}: Processor {p-1} passes check.")
    print("All checks passed.")

test_processor_steps_pandas()