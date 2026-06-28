#!/usr/bin/env python3
"""
Causality test for SwinHPClimatesetTemporalAtnCausal.

The test constructs two random inputs that are identical up to a cutoff
timestep and differ only afterwards, then verifies that outputs at or before
the cutoff are bit-for-bit identical (within floating-point tolerance).

Usage
-----
  # Defaults: nside=8, T=6, cutoff_t=3, tolerance=1e-4
  python test_causality.py

  # Larger resolution, custom split point
  python test_causality.py --nside 16 --T 8 --cutoff-t 5

  # Run on GPU
  python test_causality.py --device cuda

  # Sweep all valid cutoff points automatically
  python test_causality.py --sweep
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import torch

from experiments.climate.models.climate_pear_temporal_atn_causal import (
    SwinHPClimatesetTemporalAtnCausal,
    SwinHPClimatesetTemporalAtnCausalConfig,
    test_causality,
)
from experiments.climate.data.climateset_data_hp import ClimatesetDataSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_tiny_config(nside: int) -> SwinHPClimatesetTemporalAtnCausalConfig:
    """Return a small config suitable for fast CPU tests."""
    return SwinHPClimatesetTemporalAtnCausalConfig(
        nside=nside,
        base_pix=12,
        patch_size=4,          # small patch so N_patches is still divisible
        window_size=[2, 4],    # (D_window, N_window) — tiny windows for small nside
        shift_size=1,
        shift_strategy="nest_roll",
        depths=[2, 2, 2, 2],
        num_heads=[2, 4, 4, 2],
        embed_dims=[16, 32, 32, 16],
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,    # disable stochastic depth so the test is deterministic
    )


def build_data_spec(nside: int, T: int, n_in: int = 4, n_out: int = 2) -> ClimatesetDataSpec:
    return ClimatesetDataSpec(
        nside=nside,
        n_input_channels=n_in,
        n_output_channels=n_out,
        seq_len=T,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causality test for SwinHPClimatesetTemporalAtnCausal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nside", type=int, default=8,
                   help="HEALPix nside. Use 8 or 16 for fast CPU tests.")
    p.add_argument("--n-input-channels", type=int, default=4,
                   help="Number of input channels.")
    p.add_argument("--n-output-channels", type=int, default=2,
                   help="Number of output channels.")
    p.add_argument("--T", type=int, default=6,
                   help="Total number of timesteps in the synthetic input.")
    p.add_argument("--cutoff-t", type=int, default=3,
                   help="Split point: inputs agree for t <= cutoff_t, differ after.")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Max acceptable absolute difference in the past output window.")
    p.add_argument("--device", default="cpu",
                   help="Torch device string, e.g. 'cpu' or 'cuda'.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for reproducibility.")
    p.add_argument("--sweep", action="store_true",
                   help="Test causality at every valid cutoff point 0..T-2.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Causality test — SwinHPClimatesetTemporalAtnCausal")
    print("=" * 60)
    print(f"  nside            : {args.nside}")
    print(f"  n_input_channels : {args.n_input_channels}")
    print(f"  n_output_channels: {args.n_output_channels}")
    print(f"  T                : {args.T}")
    print(f"  device           : {args.device}")
    print(f"  tolerance        : {args.tol}")
    print()

    config = build_tiny_config(args.nside)
    data_spec = build_data_spec(
        args.nside,
        T=args.T,
        n_in=args.n_input_channels,
        n_out=args.n_output_channels,
    )

    print("Building model …")
    model = SwinHPClimatesetTemporalAtnCausal(config, data_spec)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print()

    if args.sweep:
        # Test causality at every cutoff from 0 to T-2
        cutoffs = list(range(args.T - 1))
        print(f"Sweeping cutoff_t over {cutoffs} …")
        all_passed = True
        for cutoff in cutoffs:
            torch.manual_seed(args.seed)  # reset so each cutoff uses fresh random data
            try:
                test_causality(
                    model, data_spec,
                    T=args.T,
                    cutoff_t=cutoff,
                    device=args.device,
                    tol=args.tol,
                )
                print(f"  cutoff_t={cutoff}  PASSED")
            except AssertionError as e:
                print(f"  cutoff_t={cutoff}  FAILED: {e}")
                all_passed = False

        print()
        if all_passed:
            print("All cutoff points passed. Model is causal.")
        else:
            print("One or more cutoff points FAILED. Model is NOT fully causal.")
            sys.exit(1)

    else:
        cutoff = args.cutoff_t
        if cutoff < 0 or cutoff >= args.T - 1:
            print(f"Error: --cutoff-t must be in [0, T-2] = [0, {args.T-2}].")
            sys.exit(1)

        print(f"Testing causality at cutoff_t={cutoff} …")
        try:
            test_causality(
                model, data_spec,
                T=args.T,
                cutoff_t=cutoff,
                device=args.device,
                tol=args.tol,
            )
            print()
            print(f"PASSED — outputs at t <= {cutoff} are identical within tol={args.tol}.")
            print(f"         Outputs at t > {cutoff} correctly differ between the two inputs.")
        except AssertionError as e:
            print()
            print(f"FAILED: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
