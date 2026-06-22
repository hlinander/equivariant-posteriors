"""Per-task free-rollout diagnostics for the multitask operator transformer.

DET samples: free-rollout log|det| MAE vs true.
INV samples: reduction error ||A_T - I|| and inverse error
||P - A^{-1}||/||A^{-1}|| (P = composed operator product).
Plus avg ops per task (self-chosen compute).
"""
import torch

import lib.render_duck as duck

SUBSAMPLE_N = 2048


def _run_for_split(model, state, dataset, split, seed):
    if not hasattr(dataset, "xs"):
        return
    n_total = len(dataset)
    if n_total == 0:
        return
    device = next(model.parameters()).device
    gen = torch.Generator(device=dataset.xs.device).manual_seed(int(seed))
    idx = torch.randperm(n_total, device=dataset.xs.device, generator=gen)[: min(SUBSAMPLE_N, n_total)]
    a = dataset.xs[idx].to(device)
    task = dataset.task[idx].to(device)
    logdet_true = dataset.ys[idx].to(device).squeeze(-1)
    n = a.shape[1]

    final, P, n_ops = model.free_rollout(a, task)

    def emit(name, value):
        duck.insert_checkpoint_sample_metric(
            state.model_id, state.batch, f"{name}_{split}",
            type(dataset).__name__, [], float(value), [],
        )

    det = task == 0
    inv = task == 1
    if det.any():
        diag = final[det].diagonal(dim1=1, dim2=2)
        ld = torch.log(diag.abs().clamp_min(1e-6)).sum(dim=1)
        emit("det_logdet_mae", (ld - logdet_true[det]).abs().mean().item())
        emit("det_avg_ops", n_ops[det].float().mean().item())
    if inv.any():
        recon = (final[inv] - torch.eye(n, device=device)).norm(dim=(1, 2)).mean().item()
        ainv = torch.linalg.inv(a[inv])
        inv_relerr = ((P[inv] - ainv).norm(dim=(1, 2)) / ainv.norm(dim=(1, 2)).clamp_min(1e-30)).median().item()
        emit("inv_recon_err", recon)
        emit("inv_relerr", inv_relerr)
        emit("inv_avg_ops", n_ops[inv].float().mean().item())


def compute_multi_diagnostics(model, train_run, state, device_id):
    if not (state.epoch % 25 == 0 or state.epoch >= train_run.epochs - 5):
        return
    was = model.training
    model.eval()
    seed = state.epoch * 10007 + train_run.train_config.ensemble_id
    try:
        with torch.no_grad():
            _run_for_split(model, state, state.train_dataloader.dataset, "train", seed)
            if state.val_dataloader is not None:
                _run_for_split(model, state, state.val_dataloader.dataset, "val", seed + 1)
    finally:
        if was:
            model.train()
