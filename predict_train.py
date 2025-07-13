from workload import Workload

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

scaling_factor = 1000.0
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class TemporalWorkloadPredictor(nn.Module):
    def __init__(self, H: int, W: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.H = H * 6
        self.W = W
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.Softplus(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Softplus(),
        )

        self.temporal_model = nn.LSTM(
            input_size=64 * self.H * self.W,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.1 if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256 * 12 * 12),
            nn.Softplus(),
            nn.Unflatten(1, (256, 12, 12)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Softplus(),
        )

    def forward(self, x):  # (B, T, 1, 6H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, 1, H, W)
        x = self.encoder(x)
        x = x.view(B, T, -1)
        out, _ = self.temporal_model(x)
        last = out[:, -1]
        x = self.decoder(last)
        x = nn.functional.interpolate(
            x, size=(self.H, self.W), mode="bilinear", align_corners=False
        )
        x = x.view(B, 1, self.H, self.W)
        return x


def extract_resolution(workload: Workload):
    resolution = workload.resolution
    intervals = workload.intervals
    H, W = resolution, resolution
    return H, W, intervals


def prepare_temporal_dataset(workload, sequence_length=6, train_ratio=0.8):
    H, W, T = extract_resolution(workload)
    raw = workload.workload.values.reshape(6, H, W, T).transpose(3, 0, 1, 2)

    raw = raw.reshape(T, 1, 6 * H, W)
    num_samples = T - sequence_length
    num_train = int(num_samples * train_ratio)

    x_np = np.empty((num_samples, sequence_length, 1, 6 * H, W), dtype=np.float32)
    y_np = np.empty((num_samples, 1, 6 * H, W), dtype=np.float32)

    for i in range(num_samples):
        x_np[i] = raw[i : i + sequence_length]
        y_np[i] = raw[i + sequence_length]

    x_train = torch.from_numpy(x_np[:num_train])
    y_train = torch.from_numpy(y_np[:num_train])
    x_test = torch.from_numpy(x_np[num_train:])
    y_test = torch.from_numpy(y_np[num_train:])

    y_train = y_train / scaling_factor
    y_test = y_test / scaling_factor

    return (x_train, y_train), (x_test, y_test), H, W


if __name__ == "__main__":
    res = 24
    workload_path = f"test/workloads/c{res}.csv"
    workload = Workload.read_csv(workload_path)
    (train_x, train_y), (test_x, test_y), H, W = prepare_temporal_dataset(workload)

    model = TemporalWorkloadPredictor(H, W, hidden_dim=256, num_layers=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=8, shuffle=True
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"conv_transpose2d_fullgrid_c{res}.pth")

    losses = []
    for epoch in range(50):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_x)
        scheduler.step(avg_loss)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/50 - Loss: {avg_loss:.6f}")

    torch.save({"model_state": model.state_dict(), "losses": losses}, model_path)

    # Eval
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=8)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            test_loss += criterion(pred, yb).item() * xb.size(0)
    test_loss /= len(test_x)
    print(f"Test MSE: {test_loss:.6f}")

    plt.plot(range(1, len(losses) + 1), losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()
