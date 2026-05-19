"""
Evaluate how equivariance of the first ConvNeXt block changes across training checkpoints.

For each checkpoint saved every 10 epochs:
  1. Load the full model weights from the epoch checkpoint.
  2. Extract the first encoder ConvNeXt block's inner convblock Sequential.
  3. Rotate the *input* batch, run it through reshape + convblock, then unrotate the output.
  4. Compare with the reference output (no rotation) to get equivariance error per channel.
  5. Save per-epoch results to CSV and plots.

Why a custom eval (not equivariance_error from metrics.py):
  The first ConvNeXt block maps 69 → 136 channels (with default n_channels=(136,68,34)),
  so dataset_output_reshape cannot split the output back into surface/upper tensors.
  Instead we work directly on the intermediate (B*12, C_feat, H, W) tensor,
  flatten it to (B, C_feat, Npix) using the face ordering from dataset_input_reshape,
  and apply rotate_tensor_last_dim_healpix to it directly.

Usage:
    uv run python run.py experiments/weather/evaluate_equivariance_training.py \\
        experiments/weather/persisted_configs/equivariant_ds/conv_equiv_ds.py

Optional positional args:
    argv[2]  max_epochs   (default 200)
    argv[3]  sensitivity  number of rotation angles = sensitivity-1 (default 120)
"""

import importlib
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from experiments.weather.data import DataHP, DataHPConfig
from experiments.weather.metrics import rotate_tensor_last_dim_healpix, shift_sample
from lib.ddp import ddp_setup
from lib.paths import get_model_epoch_checkpoint_path
from lib.serialization import DeserializeConfig, create_model


def load_create_config(module_file_path: str):
    module_name = Path(module_file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    return config_file.create_config


def load_model_at_epoch(create_config, epoch: int, device_id):
    """Load the full HEALPixPearConv model at a given epoch checkpoint. Returns None if missing."""
    train_run = create_config(0, epoch)
    checkpoint_path = get_model_epoch_checkpoint_path(train_run.train_config, epoch)

    if not checkpoint_path.is_file():
        print(f"  Epoch {epoch:4d}: no checkpoint at {checkpoint_path}, skipping.")
        return None

    print(f"  Epoch {epoch:4d}: loading {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device_id))
    deser_config = DeserializeConfig(train_run=train_run, device_id=device_id)
    model = create_model(deser_config, state_dict)
    model.eval()
    return model


def equivariance_error_first_layer(
    conv_model,
    dataloader,
    device,
    sensitivity: int = 120,
    nested: bool = True,
    max_batches: int = 1,
) -> dict[float, torch.Tensor]:
    """
    Measure equivariance of the first ConvNeXt block (inner convblock Sequential).

    For each rotation angle α:
      y       = convblock(reshape(x))                  # reference output, (B*12, C_feat, H, W)
      y_r     = convblock(reshape(R_α(x)))             # rotated-input output
      EE      = mean_pixels_batch |flatten(y) - R_{-α}(flatten(y_r))|

    The intermediate tensor is flattened to (B, C_feat, Npix) preserving the
    nested HEALPix face ordering used by dataset_input_reshape, so
    rotate_tensor_last_dim_healpix applies correctly.

    Returns:
        dict mapping angle_deg → tensor of shape [C_feat]
    """
    conv_model = conv_model.to(device)
    conv_model.eval()
    convblock = conv_model.encoder.encoder[0][0].convblock

    angles = [i * (360.0 / sensitivity) for i in range(1, sensitivity)]
    feature_sums: dict[float, torch.Tensor | None] = {a: None for a in angles}
    counts: dict[float, int] = {a: 0 for a in angles}

    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="batches")):
        if batch_idx >= max_batches:
            break

        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        B = batch["input_surface"].shape[0]

        # Reference forward pass
        with torch.no_grad():
            x = conv_model.dataset_input_reshape(batch)  # (B*12, 69, H, W)
            y = convblock(x)                              # (B*12, C_feat, H, W)

        C_feat, H, W = y.shape[1], y.shape[2], y.shape[3]
        Npix = 12 * H * W

        # Unflatten to (B, C_feat, Npix) preserving nested face order:
        #   (B*12, C_feat, H, W) → (B, 12, C_feat, H, W) → (B, C_feat, 12, H, W) → (B, C_feat, Npix)
        y_flat = (
            y.reshape(B, 12, C_feat, H, W)
             .permute(0, 2, 1, 3, 4)
             .reshape(B, C_feat, Npix)
        )

        for angle_deg in angles:
            rotated_batch = shift_sample(batch, angle_deg, nested=nested)

            with torch.no_grad():
                x_r = conv_model.dataset_input_reshape(rotated_batch)
                y_r = convblock(x_r)  # (B*12, C_feat, H, W)

            y_r_flat = (
                y_r.reshape(B, 12, C_feat, H, W)
                   .permute(0, 2, 1, 3, 4)
                   .reshape(B, C_feat, Npix)
            )

            # Unrotate the output so we can compare with y_flat
            y_r_unrot = rotate_tensor_last_dim_healpix(y_r_flat, -angle_deg, nested=nested)

            # MAE per channel, averaged over batch and pixels
            err = (y_flat - y_r_unrot).abs().mean(dim=(0, 2))  # [C_feat]

            if feature_sums[angle_deg] is None:
                feature_sums[angle_deg] = err.detach().clone()
            else:
                feature_sums[angle_deg] += err.detach()
            counts[angle_deg] += 1

    return {
        angle: (feature_sums[angle] / counts[angle]).cpu()
        for angle in angles
        if feature_sums[angle] is not None
    }


def _cache_path(cache_dir: str, model_name: str, epoch: int, sensitivity: int, max_batches: int) -> Path:
    tag = f"{model_name}_ep{epoch}_sens{sensitivity}_mb{max_batches}"
    return Path(cache_dir) / f"{tag}.pkl"


def _load_cache(path: Path):
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  Cache read failed ({path.name}): {e} — ignoring.")
    return None


def _save_cache(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config_file.py> [max_epochs=200] [sensitivity=120]")
        sys.exit(1)

    config_path = sys.argv[1]
    max_epochs  = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    sensitivity = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    max_batches = 1
    cache_dir   = ".local/equiv_training_cache"

    device_id   = ddp_setup()
    create_config = load_create_config(config_path)
    model_name  = Path(config_path).stem

    # One sample from 2019 test set (same as notebook)
    ds = DataHP(DataHPConfig(nside=64, start_year=2019, end_year=2019))
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    records = []

    for epoch in range(10, max_epochs + 1, 10):
        cp = _cache_path(cache_dir, model_name, epoch, sensitivity, max_batches)
        ee = _load_cache(cp)
        if ee is not None:
            print(f"  Epoch {epoch:4d}: cache hit ({cp.name})")
        else:
            model = load_model_at_epoch(create_config, epoch, device_id)
            if model is None:
                continue

            ee = equivariance_error_first_layer(
                model, dl, device=device_id, sensitivity=sensitivity, max_batches=max_batches
            )
            _save_cache(cp, ee)
            print(f"  Epoch {epoch:4d}: saved to cache ({cp.name})")

        # Summarise
        all_errors = torch.stack(list(ee.values()))  # (n_angles, C_feat)
        mean_ee = all_errors.mean().item()
        print(f"  Epoch {epoch:4d}: mean first-layer EE = {mean_ee:.5f}")

        for angle, per_channel in ee.items():
            for ch_idx, val in enumerate(per_channel.numpy()):
                records.append(dict(epoch=epoch, angle_deg=angle, channel=ch_idx, ee=float(val)))

    if not records:
        print("No checkpoints found — nothing to save.")
        sys.exit(1)

    df = pd.DataFrame(records)
    out_dir = Path(f"experiments/weather/evaluation/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "equivariance_first_layer_training.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results → {csv_path}")

    # Plot 1: mean EE (over all channels and angles) vs epoch
    summary = df.groupby("epoch")["ee"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    plt.plot(summary["epoch"], summary["ee"], marker="o")
    plt.title(f"First ConvNeXt block — mean equivariance error vs epoch\n({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Mean EE (MAE, all channels & angles)")
    plt.grid(True)
    plt.tight_layout()
    p1 = out_dir / "equivariance_first_layer_vs_epoch.png"
    plt.savefig(p1)
    plt.close()
    print(f"Saved plot → {p1}")

    # Plot 2: EE vs rotation angle for first, middle, last available epoch (channel-mean)
    epochs_available = sorted(df["epoch"].unique())
    selected = [epochs_available[0], epochs_available[len(epochs_available) // 2], epochs_available[-1]]
    angle_summary = df.groupby(["epoch", "angle_deg"])["ee"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    for ep in selected:
        sub = angle_summary[angle_summary["epoch"] == ep].sort_values("angle_deg")
        plt.plot(sub["angle_deg"], sub["ee"], marker="o", label=f"epoch {ep}")
    plt.title(f"First ConvNeXt block — EE vs rotation angle\n({model_name})")
    plt.xlabel("Rotation angle (°)")
    plt.ylabel("Mean EE (MAE, all channels)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    p2 = out_dir / "equivariance_first_layer_vs_angle.png"
    plt.savefig(p2)
    plt.close()
    print(f"Saved plot → {p2}")


if __name__ == "__main__":
    main()
