#!/usr/bin/env python
"""List available checkpoints ordered by date."""
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from lib.compute_env import env


def list_checkpoints():
    checkpoint_dir = env().paths.checkpoints
    if not checkpoint_dir.is_dir():
        print(f"No checkpoint directory at {checkpoint_dir}")
        return

    candidates = [
        cp for cp in checkpoint_dir.iterdir()
        if cp.name.startswith("checkpoint_") and (cp / "train_run.json").is_file()
    ]

    entries = []
    for cp in tqdm(candidates, desc="Scanning checkpoints"):
        hash_hex = cp.name[len("checkpoint_"):]
        json_path = cp / "train_run.json"
        mtime = cp.stat().st_mtime
        saved = json.loads(json_path.read_text())
        data = saved.get("__data__", {})
        project = data.get("project", "?")
        tc = data.get("train_config", {}).get("__data__", {})
        mc = tc.get("model_config", {})
        dc = tc.get("train_data_config", {})
        model_class = mc.get("__class__", "?") if isinstance(mc, dict) else "?"
        data_class = dc.get("__class__", "?") if isinstance(dc, dict) else "?"
        epoch_checkpoints = list(cp.glob("model_epoch_*"))
        has_latest = (cp / "model").is_file()
        if epoch_checkpoints:
            n_checkpoints = len(epoch_checkpoints)
        elif has_latest:
            n_checkpoints = 1
        else:
            n_checkpoints = 0
        entries.append((mtime, hash_hex, project, model_class, data_class, n_checkpoints))

    entries.sort(key=lambda e: e[0], reverse=True)

    print()
    print(f"{'hash':<18} {'project':<14} {'model':<28} {'data':<28} {'saves':>6}")
    print("-" * 98)
    for mtime, hash_hex, project, model_class, data_class, n_checkpoints in entries:
        date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        print(f"{hash_hex:<18} {project:<14} {model_class:<28} {data_class:<28} {n_checkpoints:>6}  {date}")


if __name__ == "__main__":
    list_checkpoints()
