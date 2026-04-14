#!/usr/bin/env python
"""
Evaluation runner for climate model variants.

Evaluates specific epochs for each model variant.
Usage: Set EVAL_EPOCHS env var (default: "200") for epochs to evaluate.
"""
import os
import sys
from pathlib import Path

# Get variant index from SLURM
variant_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
print(f"[eval_runner] Evaluating model variant {variant_idx}")

# Which epoch to evaluate, can be multiple e.g. EVAL_EPOCHS="100,150,200"
epochs_str = os.environ.get("EVAL_EPOCHS", "200")
epochs = [int(e.strip()) for e in epochs_str.split(",")]
print(f"[eval_runner] Will evaluate epochs: {epochs}")

# Paths
CONFIG_SCRIPT = "experiments/climate/persisted_configs/train_climate_baseline.py"
EVAL_SCRIPT = "experiments/climate/evaluation/eval_climate.py"

if __name__ == "__main__":
    for epoch in epochs:
        try:
            print(f"\n[eval_runner] ========================================")
            print(f"[eval_runner] Evaluating variant {variant_idx}, epoch {epoch}")
            print(f"[eval_runner] ========================================\n")
            
            # Call the evaluation script
            cmd = f"python {EVAL_SCRIPT} {CONFIG_SCRIPT} {epoch} {variant_idx}"
            print(f"[eval_runner] Running: {cmd}")
            
            exit_code = os.system(cmd)
            
            if exit_code != 0:
                print(f"[eval_runner] Evaluation failed for epoch {epoch} (exit code {exit_code})")
                print(f"[eval_runner] This might mean the checkpoint doesn't exist yet")
                continue
        
        except Exception as e:
            print(f"[eval_runner] ERROR evaluating epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n[eval_runner] Completed evaluation for variant {variant_idx}")