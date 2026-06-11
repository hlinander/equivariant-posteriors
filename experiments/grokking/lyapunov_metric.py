"""Compute top finite-time Lyapunov exponent (FTLE) for train and val data at
validation epochs.

Wired in via `TrainRun.post_validate_hook`. Subsamples N=4096 samples
deterministically (seed = epoch * 10007 + ensemble_id) per dataset, computes
log of the top singular value of the per-sample input->logits Jacobian via
`lib.lyapunov.lambda1fast`, and writes the per-split mean to the
`checkpoint_sample_metric` analytics table.
"""
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

import lib.render_duck as duck


SUBSAMPLE_N = 4096
JACOBIAN_CHUNK = 1024
METRIC_NAME = "ftle"


def _subsample_indices(n_total: int, device, seed: int) -> torch.Tensor:
    n = min(SUBSAMPLE_N, n_total)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randperm(n_total, device=device, generator=gen)[:n]


def _compute_ftle(model, xs: torch.Tensor) -> torch.Tensor:
    """Top FTLE per sample. xs shape: (B, *input_shape) — flat (B, D_in) for
    MLPs or (B, seq, D) for transformers.

    Top singular value via eigh of the small (out_dim, out_dim) Gram J J^T —
    avoids batched SVD on small matrices, which has no efficient cusolver path
    and dominates runtime by ~400x compared to this route.
    """

    def adapter(xi: torch.Tensor) -> torch.Tensor:
        # xi: (*input_shape,) single sample (vmap strips the batch dim)
        return model({"input": xi.unsqueeze(0)})["logits"].squeeze(0)

    out = []
    for start in range(0, xs.shape[0], JACOBIAN_CHUNK):
        # Fused attention backward has no vmap batching rule, so vmap falls
        # back to a serial per-sample loop; the math backend decomposes
        # attention into batchable ops (~250x faster here, no-op for MLPs).
        with sdpa_kernel([SDPBackend.MATH]):
            jac = torch.func.vmap(torch.func.jacrev(adapter))(xs[start : start + JACOBIAN_CHUNK])
        # (chunk, out_dim, *input_shape) -> (chunk, out_dim, D_in)
        jac = jac.reshape(jac.shape[0], jac.shape[1], -1)
        gram = jac @ jac.transpose(-1, -2)
        evs = torch.linalg.eigvalsh(gram)
        out.append(0.5 * torch.log(evs[:, -1].clamp_min(1e-30)))
    return torch.cat(out, dim=0)


def _run_for_split(model, state, dataset, split: str, seed_base: int):
    if not hasattr(dataset, "xs"):
        return  # not a finite-field-det-style dataset; skip
    n_total = len(dataset)
    if n_total == 0:
        return
    model_device = next(model.parameters()).device
    idx = _subsample_indices(n_total, dataset.xs.device, seed_base)
    xs = dataset.xs[idx].to(model_device)
    lambdas = _compute_ftle(model, xs)
    duck.insert_checkpoint_sample_metric(
        state.model_id,
        state.batch,
        f"{METRIC_NAME}_{split}",
        type(dataset).__name__,
        [],
        float(lambdas.detach().float().mean().item()),
        [],
    )


def compute_lyapunov_for_epoch(model, train_run, state, device_id):
    """Post-validate hook entry point."""
    was_training = model.training
    model.eval()
    seed_base = state.epoch * 10007 + train_run.train_config.ensemble_id
    try:
        # jacrev requires grad enabled; existing validate() leaves us under
        # torch.no_grad(), so re-enable explicitly here.
        with torch.enable_grad():
            _run_for_split(
                model, state, state.train_dataloader.dataset, "train", seed_base,
            )
            if state.val_dataloader is not None:
                _run_for_split(
                    model, state, state.val_dataloader.dataset, "val", seed_base + 1,
                )
    finally:
        if was_training:
            model.train()
