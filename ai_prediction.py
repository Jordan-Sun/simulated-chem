from workload import Workload

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr, pearsonr


import matplotlib.pyplot as plt

# Input shape: (batch_size, seq_len, 6, H, W)
# Output shape: (batch_size, 6, H, W) -- next time step
class TemporalWorkloadPredictor(nn.Module):
    def __init__(self, H: int, W: int):
        super().__init__()
        self.H = H
        self.W = W

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.temporal_model = nn.LSTM(64 * H * W, 256, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * H * W),
            nn.ReLU(),
            nn.Unflatten(1, (64, H, W)),
            nn.Conv2d(64, 6, kernel_size=1),
        )

    def forward(self, x):  # x: (B, T, 6, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.spatial_encoder(x)  # (B*T, 64, H, W)
        x = x.view(B, T, -1)  # (B, T, 64*H*W)
        out, _ = self.temporal_model(x)
        x = self.decoder(out[:, -1])  # Use last time step output
        return x  # (B, 6, H, W)


def extract_resolution(workload: Workload):
    # Workload shape: (6 * R * R, T)
    resolution = workload.resolution
    intervals = workload.intervals
    H, W = resolution, resolution
    return H, W, intervals


def prepare_temporal_dataset(workload, sequence_length=6, train_ratio=0.8):
    H, W, T = extract_resolution(workload)
    raw = workload.workload.values.reshape(6, H, W, T).transpose(
        3, 0, 1, 2
    )  # (T, 6, H, W)

    num_samples = T - sequence_length
    num_train = int(num_samples * train_ratio)

    x_np = np.empty((num_samples, sequence_length, 6, H, W), dtype=np.float32)
    y_np = np.empty((num_samples, 6, H, W), dtype=np.float32)

    for i in range(num_samples):
        x_np[i] = raw[i : i + sequence_length]
        y_np[i] = raw[i + sequence_length]

    # Split train/test
    x_train = torch.from_numpy(x_np[:num_train])
    y_train = torch.from_numpy(y_np[:num_train])
    x_test = torch.from_numpy(x_np[num_train:])
    y_test = torch.from_numpy(y_np[num_train:])

    return (x_train, y_train), (x_test, y_test), H, W

# Load and prepare data
workload_path = "test/workloads/c90.csv"
workload = Workload.read_csv(workload_path)
x_tensor, y_tensor, H, W = prepare_temporal_dataset(workload, sequence_length=6)

# Train model
model = TemporalWorkloadPredictor(H, W)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5
losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(loader.dataset)
    losses.append(avg_loss)

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/temporal_c90.pth")

plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()
