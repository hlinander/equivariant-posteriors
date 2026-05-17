"""
Verify that the pr_variable_weighing loss indexing is correct.

Shape convention: (B, 2, N_pix) — variables in dim 1, pixels in dim 2.
  dim 1, index 0 = tas (temperature)
  dim 1, index 1 = pr  (precipitation)
"""

import torch
import sys

B, N_pix = 4, 12288
mse = torch.nn.MSELoss()

def loss_fixed(pred, target, w):
    if w == 0.5:
        return mse(pred, target)
    loss_tas = mse(pred[:, 0], target[:, 0])
    loss_pr  = mse(pred[:, 1], target[:, 1])
    return (1.0 - w) * loss_tas + w * loss_pr

def loss_broken(pred, target, w):
    if w == 0.5:
        return mse(pred, target)
    loss_tas = mse(pred[..., 0], target[..., 0])
    loss_pr  = mse(pred[..., 1], target[..., 1])
    return (1.0 - w) * loss_tas + w * loss_pr

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(condition, msg):
    tag = PASS if condition else FAIL
    print(f"  {tag} {msg}")
    return condition

all_passed = True

# ── 1. Shape sanity ──────────────────────────────────────────────────────────
print("\n=== 1. Indexing shapes ===")
pred = torch.randn(B, 2, N_pix)

tas_fixed  = pred[:, 0]       # correct
tas_broken = pred[..., 0]     # wrong

r = check(tas_fixed.shape == (B, N_pix),
          f"pred[:, 0] covers all pixels: shape={tuple(tas_fixed.shape)}")
all_passed &= r

r = check(tas_broken.shape == (B, 2),
          f"pred[..., 0] only covers 2 values (1 pixel!): shape={tuple(tas_broken.shape)}")
all_passed &= r

# ── 2. Loss values match mse(pred, target) when weights are equal ────────────
print("\n=== 2. Fixed loss at w=0.5 matches full MSE ===")
target = torch.randn(B, 2, N_pix)

loss_half   = loss_fixed(pred, target, 0.5)
loss_equal  = loss_fixed(pred, target, 0.5)          # fast path
loss_manual = 0.5 * mse(pred[:, 0], target[:, 0]) + 0.5 * mse(pred[:, 1], target[:, 1])

r = check(abs(loss_half.item() - loss_manual.item()) < 1e-5,
          f"w=0.5 fast path ≈ equal manual split: {loss_half.item():.6f} vs {loss_manual.item():.6f}")
all_passed &= r

# ── 3. Known-value loss test ─────────────────────────────────────────────────
print("\n=== 3. Known-value loss (pred differs only in one variable) ===")
pred_known   = torch.zeros(B, 2, N_pix)
target_known = torch.zeros(B, 2, N_pix)
# Only tas is wrong (by 2.0); pr is perfect
pred_known[:, 0, :] = 2.0

expected_tas_loss = 4.0  # MSE of constant error 2.0
expected_pr_loss  = 0.0

for w in [0.1, 0.3, 0.7, 0.9]:
    loss = loss_fixed(pred_known, target_known, w)
    expected = (1 - w) * expected_tas_loss + w * expected_pr_loss
    r = check(abs(loss.item() - expected) < 1e-5,
              f"w={w}: loss={loss.item():.6f}, expected={(expected):.6f}")
    all_passed &= r

# ── 4. Gradient flows to all pixels ─────────────────────────────────────────
print("\n=== 4. Gradients reach all pixels ===")
for w in [0.1, 0.5, 0.9]:
    pred_g  = torch.randn(B, 2, N_pix, requires_grad=True)
    target_g = torch.randn(B, 2, N_pix)
    loss_fixed(pred_g, target_g, w).backward()
    grad = pred_g.grad
    tas_nonzero = (grad[:, 0, :] != 0).all()
    pr_nonzero  = (grad[:, 1, :] != 0).all()
    r = check(tas_nonzero and pr_nonzero,
              f"w={w}: grad nonzero for all pixels in both variables")
    all_passed &= r

# ── 5. Broken version has near-zero gradients for almost all pixels ──────────
print("\n=== 5. Broken version only has gradients for 2 pixels ===")
for w in [0.1, 0.9]:
    pred_g  = torch.randn(B, 2, N_pix, requires_grad=True)
    target_g = torch.randn(B, 2, N_pix)
    loss_broken(pred_g, target_g, w).backward()
    grad = pred_g.grad
    nonzero_count = (grad != 0).sum().item()
    # broken: grad only exists for pixel 0 and pixel 1 across all batches and vars
    expected_nonzero = B * 2 * 2   # B batches, 2 vars, 2 pixels touched
    r = check(nonzero_count == expected_nonzero,
              f"w={w}: broken version has grads for {nonzero_count} elements, "
              f"expected {expected_nonzero} (out of {B*2*N_pix} total)")
    all_passed &= r

# ── 6. Fixed and broken agree at w=0.5 (fast path used) ─────────────────────
print("\n=== 6. Both versions agree at w=0.5 (fast path, no indexing) ===")
pred   = torch.randn(B, 2, N_pix)
target = torch.randn(B, 2, N_pix)
r = check(abs(loss_fixed(pred, target, 0.5).item() - loss_broken(pred, target, 0.5).item()) < 1e-6,
          "w=0.5 fast path identical in both versions")
all_passed &= r

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if all_passed:
    print("All checks passed — the fix is correct.")
    sys.exit(0)
else:
    print("Some checks FAILED.")
    sys.exit(1)
