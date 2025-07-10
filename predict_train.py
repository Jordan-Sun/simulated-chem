from workload import Workload

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import time


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
    resolution = workload.resolution
    intervals = workload.intervals
    H, W = resolution, resolution
    return H, W, intervals


def prepare_temporal_dataset(workload, sequence_length=6, train_ratio=0.8):
    H, W, T = extract_resolution(workload)
    raw = workload.workload.values.reshape(6, H, W, T).transpose(3, 0, 1, 2)

    num_samples = T - sequence_length
    num_train = int(num_samples * train_ratio)

    x_np = np.empty((num_samples, sequence_length, 6, H, W), dtype=np.float32)
    y_np = np.empty((num_samples, 6, H, W), dtype=np.float32)

    for i in range(num_samples):
        x_np[i] = raw[i : i + sequence_length]
        y_np[i] = raw[i + sequence_length]

    x_train = torch.from_numpy(x_np[:num_train])
    y_train = torch.from_numpy(y_np[:num_train])
    x_test = torch.from_numpy(x_np[num_train:])
    y_test = torch.from_numpy(y_np[num_train:])

    return (x_train, y_train), (x_test, y_test), H, W

if __name__ == "__main__":
    # Load and prepare data
    workload_path = "test/workloads/c90.csv"
    workload = Workload.read_csv(workload_path)
    (train_x, train_y), (test_x, test_y), H, W = prepare_temporal_dataset(
        workload, sequence_length=6
    )

    # Train model
    model = TemporalWorkloadPredictor(H, W)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=8, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    num_epochs = 50
    losses = []
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    latest_path = os.path.join(model_dir, "temporal_c90_last.pth")

    # Resume if possible
    if os.path.exists(latest_path):
        state = torch.load(latest_path)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state["epoch"] + 1
        losses = state["losses"]
        print(f"Resuming from epoch {start_epoch}")

    end_epoch = start_epoch + num_epochs

    print(f"Training for {num_epochs} epochs starting from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_x)
        losses.append(avg_loss)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{end_epoch} - Loss: {avg_loss:.6f} - Time: {elapsed:.2f}s")

    # Final save
    torch.save(
        {
            "epoch": end_epoch - 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "losses": losses,
        },
        latest_path,
    )

    # Evaluate model
    model.eval()
    with torch.no_grad():
        preds = model(test_x.to(device))
        test_loss = criterion(preds, test_y.to(device)).item()
        print(f"Test MSE: {test_loss:.6f}")

    # Plot loss
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "training_loss.png"))
