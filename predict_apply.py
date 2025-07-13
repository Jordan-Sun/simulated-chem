import os
import torch
import pandas as pd
import numpy as np
from workload import Workload
from predict_train import TemporalWorkloadPredictor, extract_resolution, scaling_factor

# --- Config ---
model_path = "models/norm_softmax_c90_last.pth"
workload_path = "test/workloads/c90.csv"
output_csv = "softmax_c90_month.csv"
sequence_length = 6
num_predictions = 720  # 30 days hourly

# --- Load model ---
workload = Workload.read_csv(workload_path)
H, W, _ = extract_resolution(workload)
model = TemporalWorkloadPredictor(H, W)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device)["model_state"])
model.to(device)
model.eval()

# --- Prepare seed sequence ---
raw_data = workload.workload.values.reshape(6, H, W, -1).transpose(
    3, 0, 1, 2
)  # (T, 6, H, W)
seed = torch.tensor(raw_data[-sequence_length:], dtype=torch.float32).unsqueeze(
    0
)  # (1, 6, 6, H, W)


# --- Predict future ---
def predict_future(model, seed_sequence, num_predictions):
    model.eval()
    device = next(model.parameters()).device
    seed_sequence = seed_sequence.to(device)
    predictions = []

    for _ in range(num_predictions):
        with torch.no_grad():
            pred = model(seed_sequence) * scaling_factor  # Rescale prediction
            predictions.append(pred[0].cpu())

        seed_sequence = torch.cat([seed_sequence[:, 1:], pred.unsqueeze(1)], dim=1)

    return torch.stack(predictions)


# --- Run prediction ---
pred_tensor = predict_future(model, seed, num_predictions)

# --- Convert and save ---
pred_array = pred_tensor.permute(1, 2, 3, 0).reshape(6 * H * W, -1).numpy()
pred_workload = pd.DataFrame(pred_array)
pred_workload.to_csv(output_csv)
print(f"Predictions saved to {output_csv}")
