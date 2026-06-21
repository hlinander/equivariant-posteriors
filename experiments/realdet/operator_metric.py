"""Free-rollout diagnostics for the matrix-operator-token transformer.

Training is teacher-forced (delta/stop losses); the real test is the free
autoregressive rollout: emit operators, apply M=I+Delta, halt on STOP, read
log|det| from the final state. Logs free-rollout logdet MAE and the average
number of operators used (the model's self-chosen compute).
"""
import torch

import lib.render_duck as duck

SUBSAMPLE_N = 4096


def _run_for_split(model, state, dataset, split, seed):
    if not hasattr(dataset, "xs"):
        return
    n_total = len(dataset)
    if n_total == 0:
        return
    device = next(model.parameters()).device
    gen = torch.Generator(device=dataset.xs.device).manual_seed(int(seed))
    idx = torch.randperm(n_total, device=dataset.xs.device, generator=gen)[
        : min(SUBSAMPLE_N, n_total)
    ]
    a = dataset.xs[idx].to(device)
    target = dataset.ys[idx].to(device).squeeze(-1)
    logdet, n_ops, _ = model.free_rollout(a)

    def emit(name, value):
        duck.insert_checkpoint_sample_metric(
            state.model_id, state.batch, f"{name}_{split}",
            type(dataset).__name__, [], float(value), [],
        )

    emit("free_logdet_mae", (logdet - target).abs().mean().item())
    emit("free_avg_ops", n_ops.float().mean().item())


def compute_operator_diagnostics(model, train_run, state, device_id):
    # Free rollout is autoregressive (expensive); run it every ~25 epochs and
    # near the end, not every validation.
    if not (state.epoch % 25 == 0 or state.epoch >= train_run.epochs - 5):
        return
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
