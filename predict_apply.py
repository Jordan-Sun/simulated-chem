import os
import torch
import pandas as pd
import numpy as np
from workload import Workload
from predict_train import TemporalWorkloadPredictor, extract_resolution, scaling_factor

# --- Config ---
res = 24
model_path = f"models/conv_transpose2d_fullgrid_c{res}.pth"
workload_path = f"test/workloads/c{res}.csv"
output_csv = f"test/workloads/fullgrid_c{res}_month.csv"
sequence_length = 6
num_predictions = 720  # 30 days hourly

# --- Load model ---
workload = Workload.read_csv(workload_path)
H, W, _ = extract_resolution(workload)
model = TemporalWorkloadPredictor(H, W, hidden_dim=256, num_layers=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device)["model_state"])
model.to(device)
model.eval()

# --- Prepare initial seed ---
raw_data = workload.workload.values.reshape(6, H, W, -1).transpose(
    3, 0, 1, 2
)  # (T, 6, H, W)
fullgrid_data = raw_data.reshape(-1, 6 * H, W)  # (T, 6H, W)
seed_sequence = (
    torch.tensor(fullgrid_data[-sequence_length:], dtype=torch.float32)
    .unsqueeze(1)
    .to(device)
)  # (T, 1, 6H, W)


# --- Predict future ---
def predict_future(model, seed, num_predictions):
    model.eval()
    predictions = []
    for _ in range(num_predictions):
        with torch.no_grad():
            input_seq = seed.unsqueeze(0)  # (1, T, 1, 6H, W)
            pred = model(input_seq) * scaling_factor  # (1, 1, 6H, W)
            predictions.append(pred[0].cpu())  # (1, 6H, W)
        seed = torch.cat([seed[1:], pred], dim=0)  # Slide window forward
    return torch.stack(predictions)  # (T, 1, 6H, W)


# --- Run prediction ---
pred_tensor = predict_future(model, seed_sequence, num_predictions)

# --- Convert and save ---
pred_array = pred_tensor.squeeze(1).permute(1, 2, 0).reshape(6 * H * W, -1).numpy()
pred_workload = pd.DataFrame(pred_array)
pred_workload.to_csv(output_csv)
print(f"Predictions saved to {output_csv}")
