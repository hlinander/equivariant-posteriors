"""Two-axis ICL diagnostics for the operator-algebra transformer's [ICL]
sentence, both via free rollout (the model's own emitted partial operators):
  - icl_err_k{1,2,...}: query error using only the first k context examples
    (in-context-learning curve: should fall as examples accrue).
  - icl_err_t{1,2,...}: query error after t emitted rank-1 partial operators
    (test-time-compute curve: Eckart-Young => should fall monotonically).
  - icl_err / icl_avg_ops: final free-rollout error and self-chosen op count.
"""
import torch

import lib.render_duck as duck

SUBSAMPLE_N = 1024


def _run_for_split(model, state, dataset, split, seed):
    if not hasattr(dataset, "cx"):
        return
    n_total = len(dataset)
    if n_total == 0:
        return
    device = next(model.parameters()).device
    gen = torch.Generator(device=dataset.cx.device).manual_seed(int(seed))
    idx = torch.randperm(n_total, device=dataset.cx.device, generator=gen)[: min(SUBSAMPLE_N, n_total)]
    batch = dict(
        context_x=dataset.cx[idx].to(device), context_y=dataset.cy[idx].to(device),
        query_x=dataset.qx[idx].to(device),
    )
    target = dataset.qy[idx].to(device)
    k_max = batch["context_x"].shape[1]

    def emit(name, value):
        duck.insert_checkpoint_sample_metric(
            state.model_id, state.batch, name, type(dataset).__name__, [], float(value), [],
        )

    # final free rollout
    yhat, n_ops, per_step = model.icl_rollout(batch)
    emit(f"icl_err_{split}", (yhat - target).norm(dim=1).mean().item())
    emit(f"icl_avg_ops_{split}", n_ops.float().mean().item())
    # test-time-compute curve: error after t partial operators
    for t, ys in enumerate(per_step):
        emit(f"icl_err_t{t+1}_{split}", (ys - target).norm(dim=1).mean().item())
    # ICL curve: error using only first k context examples
    for k in range(1, k_max + 1):
        yk, _, _ = model.icl_rollout(batch, k=k)
        emit(f"icl_err_k{k}_{split}", (yk - target).norm(dim=1).mean().item())


def compute_icl_diagnostics(model, train_run, state, device_id):
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
