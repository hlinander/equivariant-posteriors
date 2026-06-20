"""Input->logits Jacobian diagnostics for the real-determinant task.

A single post-validate hook computes the per-sample Jacobian of the model
outputs w.r.t. the input matrix once, and derives two families of metrics:

1. ftle_{split}: the finite-time Lyapunov exponent, log of the top singular
   value of the full (2 x n*n) input->logits Jacobian. Generic sensitivity /
   chaos probe (same definition as the grokking FTLE).

2. jac_relerr_{split} / jac_cos_{split}: how close the model's learned
   sensitivity of log|det| is to the analytic gradient. By Jacobi's formula
   d log|det A| / dA = A^{-T}, so a perfectly-trained model's log|det| row of
   the Jacobian (de-standardized) should equal vec(A^{-T}). We report the
   relative Frobenius error (median over samples, since A^{-T} is heavy-tailed
   near-singular) and the cosine similarity (mean) against A^{-T}.

The log|det| output is standardized in DataRealDet (target = (logabs-mu)/sd),
so the model's logabs-row gradient is scaled by 1/sd relative to the true
A^{-T}; we multiply it back by dataset.ld_sd before comparing.
"""
import torch

import lib.render_duck as duck

SUBSAMPLE_N = 4096
JACOBIAN_CHUNK = 1024


def _subsample_indices(n_total, device, seed):
    n = min(SUBSAMPLE_N, n_total)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randperm(n_total, device=device, generator=gen)[:n]


def _per_sample_jacobian(model, xs):
    """(B, out_dim, n*n) Jacobian of logits w.r.t. flat input, in chunks."""
    def adapter(xi):
        return model({"input": xi.unsqueeze(0)})["logits"].squeeze(0)

    out = []
    for start in range(0, xs.shape[0], JACOBIAN_CHUNK):
        jac = torch.func.vmap(torch.func.jacrev(adapter))(
            xs[start : start + JACOBIAN_CHUNK]
        )
        out.append(jac)
    return torch.cat(out, dim=0)


def _metrics_from_jacobian(jac, xs, n, ld_sd):
    """Returns (ftle, jac_relerr, jac_cos) per sample.

    jac: (B, 2, n*n); column 1 of logits is standardized log|det|.
    """
    # Generic FTLE: top singular value via eigh of the small Gram J J^T.
    gram = jac @ jac.transpose(-1, -2)  # (B, 2, 2)
    evs = torch.linalg.eigvalsh(gram)
    ftle = 0.5 * torch.log(evs[:, -1].clamp_min(1e-30))

    # log|det| sensitivity vs Jacobi's A^{-T}, de-standardized.
    jac_log = jac[:, 1, :].reshape(-1, n, n) * ld_sd  # d(log|det|)/dA, raw units
    a = xs.reshape(-1, n, n)
    a_inv_t = torch.linalg.inv(a).transpose(-1, -2)

    pred = jac_log.flatten(1)
    ref = a_inv_t.flatten(1)
    ref_norm = ref.norm(dim=1).clamp_min(1e-30)
    pred_norm = pred.norm(dim=1).clamp_min(1e-30)
    relerr = (pred - ref).norm(dim=1) / ref_norm
    cos = (pred * ref).sum(dim=1) / (pred_norm * ref_norm)
    return ftle, relerr, cos


def _run_for_split(model, state, dataset, split, seed_base):
    if not hasattr(dataset, "xs"):
        return
    n_total = len(dataset)
    if n_total == 0:
        return
    n = dataset.xs.shape[1]
    n_side = int(round(n**0.5))
    model_device = next(model.parameters()).device
    idx = _subsample_indices(n_total, dataset.xs.device, seed_base)
    xs = dataset.xs[idx].to(model_device)
    ld_sd = float(dataset.ld_sd)

    ftle, relerr, cos = _metrics_from_jacobian(
        _per_sample_jacobian(model, xs), xs, n_side, ld_sd
    )

    def emit(name, value):
        duck.insert_checkpoint_sample_metric(
            state.model_id, state.batch, f"{name}_{split}",
            type(dataset).__name__, [], float(value), [],
        )

    emit("ftle", ftle.detach().float().mean().item())
    # median for the heavy-tailed relative error, mean for the bounded cosine
    emit("jac_relerr", relerr.detach().float().median().item())
    emit("jac_cos", cos.detach().float().mean().item())


def compute_realdet_jacobian_metrics(model, train_run, state, device_id):
    """Post-validate hook: FTLE + A^{-T} comparison for both splits."""
    was_training = model.training
    model.eval()
    seed_base = state.epoch * 10007 + train_run.train_config.ensemble_id
    try:
        with torch.enable_grad():
            _run_for_split(
                model, state, state.train_dataloader.dataset, "train", seed_base
            )
            if state.val_dataloader is not None:
                _run_for_split(
                    model, state, state.val_dataloader.dataset, "val", seed_base + 1
                )
    finally:
        if was_training:
            model.train()
