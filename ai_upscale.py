from workload import Workload

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr, pearsonr

# --- SRCNN Model ---
class SRCNN(nn.Module):
    def __init__(self, in_channels=6):
        super(SRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 6, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.model(x)


# --- Add Phase Channel ---
def add_phase_channel(x: torch.Tensor, t_indices: torch.Tensor) -> torch.Tensor:
    N, _, H, W = x.shape
    phases = (t_indices % 9).float() / 8
    phase_maps = phases.view(N, 1, 1, 1).expand(N, 1, H, W)
    return torch.cat([x, phase_maps], dim=1)


# --- Prepare Tensor Dataset ---
import random
import torch


import random
import torch


def prepare_dataset(
    low: Workload, high: Workload, block_size: int = 9, num_test_blocks: int = 1, seed: int = 42
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    assert low.samples == 6 * low.resolution**2
    assert high.samples == 6 * high.resolution**2

    intervals = min(low.intervals, high.intervals)
    assert (
        intervals % block_size == 0
    ), "Total intervals must be divisible by block size"

    # Trim the workloads to the same number of intervals
    low.workload = low.workload.iloc[:, :intervals]
    high.workload = high.workload.iloc[:, :intervals]

    num_blocks = intervals // block_size
    assert num_test_blocks < num_blocks, "Too many test blocks requested"

    # Randomly pick non-overlapping test blocks
    rng = random.Random(seed)
    block_indices = list(range(num_blocks))
    test_blocks = sorted(rng.sample(block_indices, num_test_blocks))

    # Compute test interval indices
    test_indices = [
        i for b in test_blocks for i in range(b * block_size, (b + 1) * block_size)
    ]
    train_indices = [i for i in range(intervals) if i not in test_indices]

    # Reshape workload to (T, 6, H, W)
    low_data = low.workload.values.reshape(6, low.resolution, low.resolution, intervals).transpose(3, 0, 1, 2)
    high_data = high.workload.values.reshape(6, high.resolution, high.resolution, intervals).transpose(3, 0, 1, 2)

    # Slice and convert to tensors
    x_train = torch.tensor(low_data[train_indices], dtype=torch.float32)
    y_train = torch.tensor(high_data[train_indices], dtype=torch.float32)
    t_train = torch.tensor(train_indices, dtype=torch.long)

    x_test = torch.tensor(low_data[test_indices], dtype=torch.float32)
    y_test = torch.tensor(high_data[test_indices], dtype=torch.float32)
    t_test = torch.tensor(test_indices, dtype=torch.long)

    print(f"[prepare_dataset] Test intervals: {test_indices}")
    return (x_train, y_train, t_train), (x_test, y_test, t_test)


# --- Training Function ---
def train_srcnn(
    model,
    x,
    y,
    t=None,
    use_phase=False,
    epochs=10,
    batch_size=8,
    output_dir="models/srcnn",
    resume=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = TensorDataset(x, y, t) if use_phase else TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    os.makedirs(output_dir, exist_ok=True)

    start_epoch = 0
    if resume:
        # Find latest checkpoint
        checkpoints = [
            f
            for f in os.listdir(output_dir)
            if f.startswith("epoch_") and f.endswith(".pth")
        ]
        if checkpoints:
            latest = max(
                checkpoints, key=lambda f: int(re.search(r"epoch_(\d+)", f).group(1))
            )
            ckpt_path = os.path.join(output_dir, latest)
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            start_epoch = int(re.search(r"epoch_(\d+)", latest).group(1))
            print(f"Resuming from {ckpt_path}, starting at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            xb, yb = batch[0].to(device), batch[1].to(device)
            tb = batch[2].to(device) if use_phase else None

            if use_phase:
                xb = add_phase_channel(xb, tb)

            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

        # Save model every epoch
        save_path = os.path.join(output_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

# --- Evaluation Function ---
def evaluate_srcnn(
    model, x_test, y_test, t_test=None, use_phase=False, checkpoint=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")

    with torch.no_grad():
        if use_phase:
            x_test = add_phase_channel(x_test, t_test)

        x_test, y_test = x_test.to(device), y_test.to(device)
        preds = model(x_test)

        mse = nn.functional.mse_loss(preds, y_test).item()
        mae = nn.functional.l1_loss(preds, y_test).item()

    print(f"\n📊 Evaluation:")
    print(f"  MSE  = {mse:.6f}")
    print(f"  MAE  = {mae:.6f}")

    # Optional: rank correlation over flattened tensors
    pred_flat = preds.cpu().numpy().flatten()
    y_flat = y_test.cpu().numpy().flatten()
    spearman = spearmanr(pred_flat, y_flat).correlation
    pearson = pearsonr(pred_flat, y_flat)[0]

    print(f"  Spearman = {spearman:.4f}")
    print(f"  Pearson  = {pearson:.4f}")

    return preds

## --- Apply Model to Upscaled Workload ---
def apply_srcnn(
    model: nn.Module,
    low_workload: Workload,
    use_phase: bool = False,
) -> Workload:
    """
    Applies the SRCNN model to the entire bilinear upscaled workload.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Prepare input tensor: (T, 6, R, R)
    T = low_workload.intervals
    R = low_workload.resolution
    low_vals = low_workload.workload.values.reshape(6, R, R, T).transpose(3, 0, 1, 2)
    x = torch.tensor(low_vals, dtype=torch.float32)

    # Optional temporal phase
    t = torch.arange(T, dtype=torch.long)
    if use_phase:
        x = add_phase_channel(x, t)

    # Model inference
    with torch.no_grad():
        preds = model(x.to(device)).cpu()  # shape: (T, 6, R', R')

    # Convert to (6 * R' * R', T)
    _, _, H, W = preds.shape
    reshaped = preds.permute(1, 2, 3, 0).contiguous().numpy().reshape(6 * H * W, T)

    return Workload(pd.DataFrame(reshaped, columns=low_workload.workload.columns))


# --- Main ---
if __name__ == "__main__":
    # Load workloads
    low_res = 24
    high_res = 48
    base = "test/workloads"
    # Check if the upscaled workload exists
    if os.path.exists(f"{base}/bilinear_c{low_res}_to_c{high_res}.csv"):
        print("Using existing upscaled workload...")
        upscaled_workload = Workload.read_csv(f"{base}/bilinear_c{low_res}_to_c{high_res}.csv")
    else:
        print("Upscaling workload...")
        low_workload = Workload.read_csv(f"{base}/c{low_res}.csv")
        upscaled_workload= low_workload.upscale(high_res, 1)
        # Store the upscaled workload for future use
        upscaled_workload.write_csv(f"{base}/bilinear_c{low_res}_to_c{high_res}.csv")

    high_workload = Workload.read_csv(f"{base}/c{high_res}.csv")

    (train_x, train_y, train_t), (test_x, test_y, test_t) = prepare_dataset(upscaled_workload, high_workload)

    use_phase = True

    print(f"Training SRCNN ({"no" if not use_phase else ""} phase)...")
    model = SRCNN(in_channels=7 if use_phase else 6)
    train_srcnn(
        model,
        train_x,
        train_y,
        t=train_t if use_phase else None,
        use_phase=use_phase,
        epochs=100,
        output_dir="models/srcnn_phase" if use_phase else "models/srcnn",
    )

    print(f"Evaluating SRCNN ({"no" if not use_phase else ""} phase)...")
    evaluate_srcnn(
        model,
        test_x,
        test_y,
        t_test=test_t if use_phase else None,
        use_phase=use_phase,
    )

    print(f"Applying SRCNN ({"no" if not use_phase else ""} phase)...")
    apply_srcnn(
        model,
        upscaled_workload,
        use_phase=use_phase,
    ).write_csv(f"{base}/srcnn{'_phase' if use_phase else ''}_c{low_res}_to_c{high_res}.csv")
