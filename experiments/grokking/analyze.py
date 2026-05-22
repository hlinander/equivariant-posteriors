#!/usr/bin/env python
"""
Post-hoc analysis of grokking checkpoints.

Loads saved epoch checkpoints, computes compression and SVD diagnostics,
outputs results as CSV.

Usage:
    uv run python experiments/grokking/analyze.py
"""

import torch
import pandas as pd
from pathlib import Path

from lib.paths import get_checkpoint_path, get_model_epoch_checkpoint_path
from lib.serialization import instantiate_model
from experiments.grokking.diagnostics import (
    compressed_size,
    weight_statistics,
    layer_svd_summary,
)


def analyze_run(train_config, epochs_to_check, device="cpu"):
    checkpoint_path = get_checkpoint_path(train_config)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    model = instantiate_model(train_config)
    model.to(device)

    rows = []
    for epoch in epochs_to_check:
        ckpt_path = get_model_epoch_checkpoint_path(train_config, epoch)
        if not ckpt_path.is_file():
            continue

        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        stats = weight_statistics(model)
        csize = compressed_size(state_dict)
        svd = layer_svd_summary(model)

        avg_effective_rank = sum(l["effective_rank"] for l in svd) / len(svd) if svd else 0

        rows.append(
            dict(
                epoch=epoch,
                compressed_bytes=csize,
                l1_norm=stats["l1_norm"],
                l2_norm=stats["l2_norm"],
                near_zero_fraction=stats["near_zero_fraction"],
                avg_effective_rank=avg_effective_rank,
            )
        )
        print(f"  epoch {epoch}: compressed={csize} bytes, l2={stats['l2_norm']:.4f}, "
              f"eff_rank={avg_effective_rank:.2f}")

    return pd.DataFrame(rows) if rows else None


if __name__ == "__main__":
    # Example: analyze a specific parity run
    from experiments.grokking.parity import create_config

    config = create_config(
        width=256, depth=2, k=3, train_size=1024, weight_decay=1e-2, lr=1e-3, ensemble_id=0
    )
    train_config = config.train_config

    epochs = list(range(0, 100001, 5000))
    print(f"Analyzing checkpoints at epochs: {epochs}")
    df = analyze_run(train_config, epochs)
    if df is not None:
        out_path = Path(__file__).parent / "analysis_results.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
    else:
        print("No checkpoints found. Train the model first.")
