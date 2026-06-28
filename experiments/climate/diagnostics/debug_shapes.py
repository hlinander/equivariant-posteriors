#!/usr/bin/env python
"""
Shape debug script for SwinHPClimateset.

Traces tensor shapes from a synthetic batch (matching the HP dataset format
for seq_len=1) through every stage: dataloader → model internals → logits →
loss function.  Does NOT require the actual climate data files.

Run from the repo root:
    python experiments/climate/debug_shapes.py
"""
import torch

from experiments.climate.models.swin_hp_climateset import (
    SwinHPClimatesetConfig,
    SwinHPClimateset,
)
from experiments.climate.data.climateset_data_hp import ClimatesetDataSpec

# ---------------------------------------------------------------------------
# Config — mirrors train_climate_pear_multiseed.py exactly
# ---------------------------------------------------------------------------
NSIDE   = 32
N_PIX   = 12 * NSIDE ** 2   # 12288 for nside=32
N_IN    = 4                  # BC, CH4, SO2, CO2
N_OUT   = 2                  # tas (idx 0), pr (idx 1)
BATCH   = 2

model_config = SwinHPClimatesetConfig(
    base_pix=12,
    nside=NSIDE,
    dev_mode=False,
    depths=[2, 6, 6, 2],
    num_heads=[6, 12, 12, 6],
    embed_dims=[192 // 4, 384 // 4, 384 // 4, 192 // 4],
    window_size=[1, 64],
    use_cos_attn=False,
    use_v2_norm_placement=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0,
    rel_pos_bias="single",
    shift_size=4,
    shift_strategy="ring_shift",
    ape=False,
    patch_size=16,
)

data_spec = ClimatesetDataSpec(
    nside=NSIDE,
    n_input_channels=N_IN,
    n_output_channels=N_OUT,
    seq_len=1,
)

model = SwinHPClimateset(model_config, data_spec).eval()

# ---------------------------------------------------------------------------
# Synthetic batch — matches HP dataset __getitem__ output for seq_len=1.
# Dataset returns (C, N_pix); DataLoader stacks to (B, C, N_pix).
# ---------------------------------------------------------------------------
x      = torch.randn(BATCH, N_IN,  N_PIX)
target = torch.randn(BATCH, N_OUT, N_PIX)
batch  = {"input": x, "target": target}

print("=" * 65)
print("1. DATALOADER OUTPUT  (seq_len=1, before model)")
print(f"   batch['input'].shape        = {batch['input'].shape}   # (B, C_in, N_pix)")
print(f"   batch['target'].shape       = {batch['target'].shape}  # (B, C_out, N_pix)")
print(f"   target[:, 0, :].shape (tas) = {batch['target'][:, 0, :].shape}")
print(f"   target[:, 1, :].shape (pr)  = {batch['target'][:, 1, :].shape}")
print()

# ---------------------------------------------------------------------------
# Patch _forward to print each internal shape without touching the source file
# ---------------------------------------------------------------------------
def _debug_forward(x_surface):
    print("   [_forward] input x_surface         :", x_surface.shape, "# (B, C_in, N_pix)")
    x = model.patch_embed(x_surface)
    print("   [_forward] after patch_embed        :", x.shape, "# (B, D, N_patches, C)")
    x = model.layers[0](x)
    print("   [_forward] after layer[0] encoder   :", x.shape)
    skip = x
    x = model.downsample(x)
    print("   [_forward] after downsample         :", x.shape, "# N_patches // 4")
    x = model.layers[1](x)
    x = model.layers[2](x)
    print("   [_forward] after layers[1,2] bottl. :", x.shape)
    x = model.norm(x)
    x = model.upsample(x)
    print("   [_forward] after upsample           :", x.shape)
    x = model.layers[3](x)
    print("   [_forward] after layer[3] decoder   :", x.shape)
    x = torch.concatenate([skip, x], dim=-1)
    print("   [_forward] after skip concat        :", x.shape, "# dim doubled")
    x_surface = model.final_up(x)
    print("   [_forward] after final_up           :", x_surface.shape, "# (B, N_pix, C_out)")
    return x_surface

model._forward = _debug_forward

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
print("=" * 65)
print("2. MODEL FORWARD PASS  (SwinHPClimateset.forward)")
with torch.no_grad():
    output = model(batch)

pred = output["logits_output"]

print()
print("=" * 65)
print("3. MODEL OUTPUT")
print(f"   logits_output.shape           = {pred.shape}  # (B, C_out, N_pix)")
print(f"   pred[:, 0, :].shape  (tas)    = {pred[:, 0, :].shape}  # all tas pixels")
print(f"   pred[:, 1, :].shape  (pr)     = {pred[:, 1, :].shape}  # all pr  pixels")
print()
print("   NOTE: pred[:, 0] and pred[:, 1] are shorthand for pred[:, 0, :] and pred[:, 1, :]")
print(f"   pred[:, 0].shape              = {pred[:, 0].shape}")
print(f"   pred[:, 1].shape              = {pred[:, 1].shape}")
print()

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
mse = torch.nn.MSELoss()

print("=" * 65)
print("4. LOSS FUNCTION")
loss_full = mse(pred, target)
print(f"   mse(pred, target)                        = {loss_full.item():.4f}  (equal weight)")

loss_tas = mse(pred[:, 0], target[:, 0])
loss_pr  = mse(pred[:, 1], target[:, 1])
print(f"   mse(pred[:,0], target[:,0])  [tas]       = {loss_tas.item():.4f}")
print(f"   mse(pred[:,1], target[:,1])  [pr]        = {loss_pr.item():.4f}")

for pr_w in [0.1, 0.5, 0.9]:
    w = (1 - pr_w) * loss_tas + pr_w * loss_pr
    print(f"   weighted loss (pr_variable_weighing={pr_w}) = {w.item():.4f}")
print()

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
print("=" * 65)
print("5. SANITY CHECKS")

assert pred.shape == target.shape, \
    f"FAIL: pred {pred.shape} != target {target.shape}"
print(f"   pred.shape == target.shape          ✓  {pred.shape}")

assert pred.shape == (BATCH, N_OUT, N_PIX), \
    f"FAIL: expected ({BATCH}, {N_OUT}, {N_PIX}), got {pred.shape}"
print(f"   pred shape is (B, C_out, N_pix)     ✓  ({BATCH}, {N_OUT}, {N_PIX})")

assert pred[:, 0].shape == (BATCH, N_PIX), \
    f"FAIL: tas indexing gives {pred[:, 0].shape}"
assert pred[:, 1].shape == (BATCH, N_PIX), \
    f"FAIL: pr  indexing gives {pred[:, 1].shape}"
print(f"   pred[:,0] (tas) = (B, N_pix)        ✓  {pred[:, 0].shape}")
print(f"   pred[:,1] (pr)  = (B, N_pix)        ✓  {pred[:, 1].shape}")

# Check the forward comment (it says B, N_pix, C_surface — but that's WRONG)
# The actual logits_output is (B, C_out, N_pix), which is what we want.
assert pred.shape[1] == N_OUT, \
    f"FAIL: dim 1 should be C_out={N_OUT}, got {pred.shape[1]}"
assert pred.shape[2] == N_PIX, \
    f"FAIL: dim 2 should be N_pix={N_PIX}, got {pred.shape[2]}"
print(f"   dim 1 = C_out ({N_OUT}), dim 2 = N_pix ({N_PIX})  ✓")
print()
print("All checks passed.")
