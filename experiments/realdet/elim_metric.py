"""Post-validate diagnostics for the elimination-rollout model.

Separates the two failure modes discussed in the design:
  - det_conservation_{split}: |log|det(final)| - log|det(A)||, via slogdet of
    the rolled-out matrix. Should be ~0 by construction (every step is
    det-preserving); a sanity check that the rollout isn't corrupting the target.
  - oracle_logdet_mae_{split}: log|det| MAE of the analytic-multiplier rollout
    (Gaussian elimination without pivoting) -- the achievable ceiling / floor on
    error for this fixed-order scheme (it diverges on small pivots).
  - resid_final_{split}: mean |A[i,k]| of the last eliminated entry, a scalar
    proxy for how far the learned rollout drifted from triangular by the end.

These land in checkpoint_sample_metric alongside the model's own logdet_mae /
lower_rms epoch metrics, so the drift story is visible in the analytics.
"""
import torch

import lib.render_duck as duck

SUBSAMPLE_N = 4096


def _subsample(n_total, device, seed):
    n = min(SUBSAMPLE_N, n_total)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randperm(n_total, device=device, generator=gen)[:n]


def _run_for_split(model, state, dataset, split, seed):
    if not hasattr(dataset, "xs"):
        return
    n_total = len(dataset)
    if n_total == 0:
        return
    device = next(model.parameters()).device
    idx = _subsample(n_total, dataset.xs.device, seed)
    a = dataset.xs[idx].to(device)
    target = dataset.ys[idx].to(device).squeeze(-1)

    logdet_m, _, final_m, resid_m, mult_loss, pivot_loss = model.rollout(
        a, oracle=False, collect_resid=True
    )
    logdet_o, _, _, _, _, _ = model.rollout(a, oracle=True)
    _, logdet_final = torch.linalg.slogdet(final_m)

    def emit(name, value):
        duck.insert_checkpoint_sample_metric(
            state.model_id, state.batch, f"{name}_{split}",
            type(dataset).__name__, [], float(value), [],
        )

    emit("det_conservation", (logdet_final - target).abs().mean().item())
    emit("oracle_logdet_mae", (logdet_o - target).abs().mean().item())
    emit("mult_loss", float(mult_loss))  # per-step multiplier error (free rollout)
    if model.config.pivot == "learned":
        emit("pivot_loss", float(pivot_loss))  # pivot CE vs partial-pivot argmax
    if resid_m:
        emit("resid_final", resid_m[-1].item())
    # The test-time-compute scaling curve (logdet_mae vs refinement R) is
    # computed post-hoc on the final checkpoint (scaling_curve.py) rather than
    # in-training -- the in-hook R-sweep was far too expensive for the
    # per-step-re-encoding transformer step.


def compute_elim_diagnostics(model, train_run, state, device_id):
    was_training = model.training
    model.eval()
    seed = state.epoch * 10007 + train_run.train_config.ensemble_id
    try:
        with torch.no_grad():
            _run_for_split(model, state, state.train_dataloader.dataset, "train", seed)
            if state.val_dataloader is not None:
                _run_for_split(
                    model, state, state.val_dataloader.dataset, "val", seed + 1
                )
    finally:
        if was_training:
            model.train()
