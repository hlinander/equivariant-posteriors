#!/usr/bin/env python
"""Stage-1 elimination-rollout experiment: learn log|det| of real NxN matrices
by a det-preserving sequence of elementary row operations (fixed order, learned
multipliers), reading off Sum_d log|diag|.

Loss = L1 on log|det| (the readout) + lam * lower-tri mass (forces genuine
triangularization, so the model can't cheat the fixed readout). Metrics:
logdet_mae and lower_rms; a post-validate hook logs the oracle ceiling,
det-conservation sanity, and end-of-rollout drift.
"""
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.train_dataclasses import TrainEval
from lib.metric import Metric
from lib.models.elimination_rollout import EliminationRolloutConfig
from lib.datasets.real_det_matrix import DataRealDetMatrixConfig
from lib.generic_ablation import get_config_grid
from lib.distributed_trainer import distributed_train
from experiments.realdet.elim_metric import compute_elim_diagnostics


def logdet_mae(output, batch):
    return (output["logits"].squeeze(-1) - batch["target"].squeeze(-1)).abs()


def lower_rms(output, batch):
    return output["lower_tri_sq"].clamp_min(0).sqrt()


def _metric_list():
    return [lambda: Metric(logdet_mae), lambda: Metric(lower_rms)]


def _make_train_run(
    n, n_train, hidden, depth, lr, weight_decay, seed, lam, epochs,
    input_features="raw", multiplier_param="linear", teacher_mode="off",
    pivot="none", refine_steps=0, step_arch="mlp",
):
    # Closure so the loss hashes by __name__ (module-path-independent); lam is
    # captured, so keep it fixed across the sweep to avoid hash collisions.
    def elim_loss(output, batch):
        pred = output["logits"].squeeze(-1)
        tgt = batch["target"].squeeze(-1)
        readout = (pred - tgt).abs().mean()
        triangularize = output["lower_tri_sq"].mean()
        # per-step multiplier supervision against the oracle (the division fix
        # is verified by this term going to ~0); dominant signal under teacher
        # forcing, where the readout is exact regardless of the model.
        step = output["mult_loss"]
        # pivot classification supervision vs partial-pivot argmax (0 unless
        # pivot="learned").
        pivot_sup = output["pivot_loss"]
        return readout + lam * triangularize + step + pivot_sup

    train_eval = TrainEval(
        train_metrics=_metric_list(),
        validation_metrics=_metric_list(),
        log_gradient_norm=True,
        log_parameter_norm=True,
        log_sample_ids=False,
        diagnostics_interval=100,
    )
    train_data = DataRealDetMatrixConfig(n=n, n_train=n_train, seed=seed)
    val_data = DataRealDetMatrixConfig(n=n, n_train=n_train, seed=seed, validation=True)

    train_config = TrainConfig(
        model_config=EliminationRolloutConfig(
            hidden=hidden, depth=depth, input_features=input_features,
            multiplier_param=multiplier_param, teacher_mode=teacher_mode,
            pivot=pivot, refine_steps=refine_steps, step_arch=step_arch,
        ),
        train_data_config=train_data,
        val_data_config=val_data,
        loss=elim_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(lr=lr, weight_decay=weight_decay),
        ),
        batch_size=1024,
        gradient_clipping=1.0,
        ensemble_id=seed,
        _version=1,
    )

    return TrainRun(
        project="real_det_elim",
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epochs,
        save_nth_epoch=50,
        validate_nth_epoch=5,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=50,
        visualize_interval_s=5,
        post_validate_hook=compute_elim_diagnostics,
    )


def create_elim_config(
    n, hidden, depth, lr, weight_decay, seed, input_features,
    multiplier_param, teacher_mode, pivot, refine_steps=0, step_arch="mlp",
):
    return _make_train_run(
        n=n, n_train=100000, hidden=hidden, depth=depth, lr=lr,
        weight_decay=weight_decay, seed=seed, lam=1.0, epochs=200,
        input_features=input_features, multiplier_param=multiplier_param,
        teacher_mode=teacher_mode, pivot=pivot, refine_steps=refine_steps,
        step_arch=step_arch,
    )


def elimination_configs():
    """Stage 1: input featurization x multiplier output param, free rollout."""
    return get_config_grid(
        create_elim_config,
        dict(
            n=[2, 3, 4],
            hidden=[256], depth=[2], lr=[1e-3], weight_decay=[0.0], seed=[0, 1],
            input_features=["raw", "log", "both"],
            multiplier_param=["linear"],
            teacher_mode=["off"],
            pivot=["none"],
        ),
    )


def teacher_configs():
    """Stage 2: the division fix (log-output) + teacher forcing / scheduled
    sampling. Compares linear vs log output head under teacher_mode on/anneal,
    fixed input_features=both. Per-step mult_loss -> ~0 means division solved;
    free-rollout (eval) logdet_mae/lower_rms then isolate drift."""
    return get_config_grid(
        create_elim_config,
        dict(
            n=[2, 3, 4],
            hidden=[256], depth=[2], lr=[1e-3], weight_decay=[0.0], seed=[0, 1],
            input_features=["both"],
            multiplier_param=["linear", "log"],
            teacher_mode=["on", "anneal"],
            pivot=["none"],
        ),
    )


def pivoting_configs():
    """Stage 3: pivoting. Fixed winners (log output, anneal teacher, both
    input); compare pivot none/partial/learned across N up to 5. Partial
    pivoting bounds |c|<=1 (fixes the heavy-tailed multiplier targets that grew
    with N); learned pivoting is the discrete strategy, supervised vs the
    partial-pivot argmax (pivot_loss). Question: does pivoting finally bring
    free-rollout logdet_mae to oracle level at N>=3, and does the learned pivot
    match partial?"""
    return get_config_grid(
        create_elim_config,
        dict(
            n=[3, 4, 5],
            hidden=[256], depth=[2], lr=[1e-3], weight_decay=[0.0], seed=[0, 1],
            input_features=["both"],
            multiplier_param=["log"],
            teacher_mode=["anneal"],
            pivot=["none", "partial", "learned"],
        ),
    )


def refine_configs():
    """Stage 4: greedy residual refinement (test-time compute). Train with one
    extra sweep of refinement (refine_steps = base = n(n-1)/2); the hook logs
    free-rollout logdet_mae across an eval R-grid {0,1,2,4,8}x base, the
    inference-compute scaling curve. Fixed winners: partial pivot, log output,
    anneal teacher, both input."""
    cfgs = []
    for n in (3, 4, 5):
        base = n * (n - 1) // 2
        for seed in (0, 1):
            cfgs.append(
                lambda n=n, seed=seed, base=base: create_elim_config(
                    n=n, hidden=256, depth=2, lr=1e-3, weight_decay=0.0, seed=seed,
                    input_features="both", multiplier_param="log",
                    teacher_mode="anneal", pivot="partial", refine_steps=base,
                )
            )
    return cfgs


def arch_configs():
    """Stage 5: row-token transformer step vs MLP step. Fixed winners (partial
    pivot, log output, anneal teacher, both input, one refine sweep); the test
    is whether attention's clean entry/row selection drops mult_loss below the
    MLP's ~0.19 N=4 floor and lets refinement contract. N in {3,4,5}, 2 seeds."""
    cfgs = []
    for n in (3, 4, 5):
        base = n * (n - 1) // 2
        for arch in ("mlp", "transformer"):
            for seed in (0, 1):
                cfgs.append(
                    lambda n=n, base=base, arch=arch, seed=seed: create_elim_config(
                        n=n, hidden=256, depth=2, lr=1e-3, weight_decay=0.0, seed=seed,
                        input_features="both", multiplier_param="log",
                        teacher_mode="anneal", pivot="partial", refine_steps=base,
                        step_arch=arch,
                    )
                )
    return cfgs


def smoke_configs():
    return [
        lambda: create_elim_config(
            n=3, hidden=128, depth=2, lr=1e-3, weight_decay=0.0, seed=0,
            input_features="both", multiplier_param="log", teacher_mode="anneal",
            pivot="partial", refine_steps=3, step_arch="transformer",
        )
    ]


def create_configs():
    return elimination_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        distributed_train(smoke_configs())
    else:
        distributed_train(create_configs())
