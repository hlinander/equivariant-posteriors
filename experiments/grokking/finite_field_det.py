#!/usr/bin/env python
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.classification_metrics import create_classification_metrics
from lib.models.grok_mlp import GrokMLPConfig
from lib.models.transformer import TransformerConfig
from lib.models.transformer_encoder import TransformerEncoderConfig
from lib.datasets.finite_field_det import DataFiniteFieldDetConfig
from lib.generic_ablation import get_config_grid
from lib.optimizers import adamw_no_decay_norm_bias
from lib.distributed_trainer import distributed_train
from experiments.grokking.lyapunov_metric import compute_lyapunov_for_epoch


def _make_train_run(
    model_config,
    n,
    p,
    frac,
    weight_decay,
    lr,
    ensemble_id,
    seq=False,
    optimizer=torch.optim.AdamW,
    gradient_clipping=None,
    z_loss=False,
):
    loss = torch.nn.CrossEntropyLoss()

    def ce_loss(output, batch):
        return loss(output["logits"], batch["target"])

    def ce_z_loss(output, batch):
        # z-loss penalizes the log-partition magnitude (logsumexp), capping
        # logit growth at the source instead of letting cross-entropy margin
        # maximization migrate scale into the un-decayed LayerNorm gains.
        # lambda=1e-4 is the PaLM/T5 default. Distinct __name__ from ce_loss,
        # so this is a distinct run identity (see lib.stable_hash.json_default).
        logits = output["logits"]
        z = torch.logsumexp(logits, dim=-1)
        return loss(logits, batch["target"]) + 1e-4 * (z**2).mean()

    loss_fn = ce_z_loss if z_loss else ce_loss

    train_eval = create_classification_metrics(None, p)
    train_eval.log_gradient_norm = True
    train_eval.log_parameter_norm = True
    train_eval.log_sample_ids = False
    train_eval.diagnostics_interval = 100

    train_data = DataFiniteFieldDetConfig(
        n=n, p=p, frac=frac, seed=ensemble_id, seq=seq
    )
    val_data = DataFiniteFieldDetConfig(
        n=n, p=p, frac=frac, seed=ensemble_id, seq=seq, validation=True
    )

    train_config = TrainConfig(
        model_config=model_config,
        train_data_config=train_data,
        val_data_config=val_data,
        loss=loss_fn,
        optimizer=OptimizerConfig(
            optimizer=optimizer,
            kwargs=dict(lr=lr, weight_decay=weight_decay),
        ),
        batch_size=train_data.n_samples,
        gradient_clipping=gradient_clipping,
        ensemble_id=ensemble_id,
        _version=1,
    )

    return TrainRun(
        project="grokking_det",
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=100000,
        save_nth_epoch=1000,
        validate_nth_epoch=10,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=5000,
        visualize_interval_s=5,
        post_validate_hook=compute_lyapunov_for_epoch,
    )


def create_mlp_config(width, depth, n, p, frac, weight_decay, lr, ensemble_id):
    model_config = GrokMLPConfig(width=width, depth=depth, activation="relu")
    return _make_train_run(
        model_config, n, p, frac, weight_decay, lr, ensemble_id, seq=False
    )


def create_transformer_config(
    embed_d, num_layers, num_heads, n, p, frac, weight_decay, lr, ensemble_id
):
    model_config = TransformerConfig(
        embed_d=embed_d,
        mlp_dim=embed_d * 4,
        num_layers=num_layers,
        num_heads=num_heads,
        softmax=True,
        activation="relu",
    )
    return _make_train_run(
        model_config, n, p, frac, weight_decay, lr, ensemble_id, seq=True
    )


def create_transformer_encoder_config(
    embed_d, num_layers, num_heads, n, p, frac, weight_decay, lr, ensemble_id
):
    """Legacy overrides (post-LN, sqrt(32) embedding scale) to match the
    cross-attending Transformer runs, isolating the architecture change."""
    model_config = TransformerEncoderConfig(
        embed_d=embed_d,
        mlp_dim=embed_d * 4,
        num_layers=num_layers,
        num_heads=num_heads,
        softmax=True,
        activation="relu",
        norm_first=False,
        legacy_embed_scale=True,
    )
    return _make_train_run(
        model_config, n, p, frac, weight_decay, lr, ensemble_id, seq=True
    )


def create_transformer_encoder_stable_config(
    embed_d, num_layers, num_heads, n, p, frac, weight_decay, lr, ensemble_id
):
    """TransformerEncoder with the stability fixes: model defaults
    (pre-LN, sqrt(embed_d) embedding scale) plus no weight decay on
    normalization params / biases and gradient clipping.
    """
    model_config = TransformerEncoderConfig(
        embed_d=embed_d,
        mlp_dim=embed_d * 4,
        num_layers=num_layers,
        num_heads=num_heads,
        softmax=True,
        activation="relu",
    )
    return _make_train_run(
        model_config,
        n,
        p,
        frac,
        weight_decay,
        lr,
        ensemble_id,
        seq=True,
        optimizer=adamw_no_decay_norm_bias,
        gradient_clipping=1.0,
    )


def create_transformer_encoder_zloss_config(
    embed_d, num_layers, num_heads, n, p, frac, weight_decay, lr, ensemble_id
):
    """Stable TransformerEncoder plus z-loss. Identical to
    create_transformer_encoder_stable_config except for the logit-norm penalty,
    to test whether capping logit growth tames the post-grok val-accuracy
    collapses without losing the stable arm's fast convergence.
    """
    model_config = TransformerEncoderConfig(
        embed_d=embed_d,
        mlp_dim=embed_d * 4,
        num_layers=num_layers,
        num_heads=num_heads,
        softmax=True,
        activation="relu",
    )
    return _make_train_run(
        model_config,
        n,
        p,
        frac,
        weight_decay,
        lr,
        ensemble_id,
        seq=True,
        optimizer=adamw_no_decay_norm_bias,
        gradient_clipping=1.0,
        z_loss=True,
    )


def minimal_configs():
    """Tiny tasks to verify generalization is possible at all."""
    mlp = get_config_grid(
        create_mlp_config,
        dict(
            width=[256],
            depth=[2],
            n=[2],
            p=[2, 3],
            frac=[0.5, 0.75],
            weight_decay=[10.0, 1.0, 0.1],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )
    transformer = get_config_grid(
        create_transformer_config,
        dict(
            embed_d=[64],
            num_layers=[2],
            num_heads=[4],
            n=[2],
            p=[2, 3],
            frac=[0.5, 0.75],
            weight_decay=[1.0, 0.1],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )
    return mlp + transformer


def main_configs():
    mlp = get_config_grid(
        create_mlp_config,
        dict(
            width=[512],
            depth=[3, 4, 6],
            n=[2],
            p=[5],
            frac=[0.1, 0.3, 0.5, 0.7],
            weight_decay=[10.0, 1.0, 0.1, 0.3],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )
    transformer = get_config_grid(
        create_transformer_config,
        dict(
            embed_d=[64, 128],
            num_layers=[2, 4],
            num_heads=[4],
            n=[2],
            p=[5],
            frac=[0.1, 0.3, 0.5, 0.7],
            weight_decay=[1.0, 0.1],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )
    return mlp + transformer


def minimal_transformer_configs():
    """Transformer-only minimal configs."""
    return get_config_grid(
        create_transformer_config,
        dict(
            embed_d=[64],
            num_layers=[2],
            num_heads=[4],
            n=[2],
            p=[13, 2, 3],
            frac=[0.9, 0.3],
            weight_decay=[0.1],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )


def p13_sweep_configs():
    """Sweep over p=13 n=2 det with both MLP and transformer."""
    mlp = get_config_grid(
        create_mlp_config,
        dict(
            width=[256],
            depth=[2, 4],
            n=[2],
            p=[13],
            frac=[0.9, 0.7, 0.5, 0.3],
            weight_decay=[1.0, 0.3, 0.1],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )
    transformer = get_config_grid(
        create_transformer_config,
        dict(
            embed_d=[64],
            num_layers=[2, 4],
            num_heads=[4],
            n=[2],
            p=[13],
            frac=[0.9, 0.7, 0.5, 0.3],
            weight_decay=[1.0, 0.3, 0.1],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )
    return mlp + transformer


def p13_transformer_encoder_configs():
    """Self-attention-only transformer (TransformerEncoder) on p=13 n=2 det,
    mirroring the transformer grid in p13_sweep_configs for comparison with the
    cross-attending Transformer runs.
    """
    return get_config_grid(
        create_transformer_encoder_config,
        dict(
            embed_d=[64],
            num_layers=[2, 4],
            num_heads=[4],
            n=[2],
            p=[13],
            frac=[0.9, 0.7, 0.5, 0.3],
            weight_decay=[1.0, 0.3, 0.1],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )


def p13_transformer_encoder_stable_configs():
    """p13_transformer_encoder_configs grid with the stability fixes
    (see create_transformer_encoder_stable_config) for comparison against the
    unfixed TransformerEncoder baseline.
    """
    return get_config_grid(
        create_transformer_encoder_stable_config,
        dict(
            embed_d=[64],
            num_layers=[2, 4],
            num_heads=[4],
            n=[2],
            p=[13],
            frac=[0.9, 0.7, 0.5, 0.3],
            weight_decay=[1.0, 0.3, 0.1],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )


def p13_transformer_encoder_zloss_configs():
    """p13_transformer_encoder_stable_configs grid plus z-loss, to compare
    post-grok stability and parameter-norm growth against the no-penalty
    stable arm on the identical grid.
    """
    return get_config_grid(
        create_transformer_encoder_zloss_config,
        dict(
            embed_d=[64],
            num_layers=[2, 4],
            num_heads=[4],
            n=[2],
            p=[13],
            frac=[0.9, 0.7, 0.5, 0.3],
            weight_decay=[1.0, 0.3, 0.1],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )


def large_p_sweep_configs():
    """MLP sweep over larger primes p in {17, 23, 31}, n=2."""
    return get_config_grid(
        create_mlp_config,
        dict(
            width=[256],
            depth=[2, 4],
            n=[2],
            p=[17, 23, 31],
            frac=[0.3, 0.5, 0.7],
            weight_decay=[1.0, 0.1],
            lr=[1e-3],
            ensemble_id=[0],
        ),
    )


def cancelled_middle_configs():
    """Re-run only the 13 middle MLP configs killed by the GPU util watcher in
    job 16680904 (depth=2 p=23/31 frac=0.5/0.7 + p=31 frac=0.3, and depth=4 p=23
    frac=0.5/0.7). Same model/data identity as large_p_sweep_configs so runs
    resume from existing checkpoints.
    """
    specs = [
        (2, 23, 0.5, 1.0), (2, 23, 0.5, 0.1),
        (2, 23, 0.7, 1.0), (2, 23, 0.7, 0.1),
        (2, 31, 0.3, 1.0), (2, 31, 0.3, 0.1),
        (2, 31, 0.5, 1.0), (2, 31, 0.5, 0.1),
        (2, 31, 0.7, 1.0),
        (4, 23, 0.5, 1.0), (4, 23, 0.5, 0.1),
        (4, 23, 0.7, 1.0), (4, 23, 0.7, 0.1),
    ]
    return [
        (lambda depth=depth, p=p, frac=frac, wd=wd: create_mlp_config(
            width=256, depth=depth, n=2, p=p, frac=frac,
            weight_decay=wd, lr=1e-3, ensemble_id=0,
        ))
        for (depth, p, frac, wd) in specs
    ]


def p17_lyapunov_smoke_configs():
    """Tiny p=17 smoke set at ensemble_id=1 to validate the Lyapunov hook."""
    specs = [
        (2, 0.3, 1.0),
        (2, 0.5, 0.1),
    ]
    return [
        (lambda depth=depth, frac=frac, wd=wd: create_mlp_config(
            width=256, depth=depth, n=2, p=17, frac=frac,
            weight_decay=wd, lr=1e-3, ensemble_id=1,
        ))
        for (depth, frac, wd) in specs
    ]


def p23_p31_lyapunov_configs():
    """Full MLP grid at ensemble_id=1 with the live mean-FTLE hook, p in {23, 31}."""
    return get_config_grid(
        create_mlp_config,
        dict(
            width=[256],
            depth=[2, 4],
            n=[2],
            p=[23, 31],
            frac=[0.3, 0.5, 0.7],
            weight_decay=[1.0, 0.1],
            lr=[1e-3],
            ensemble_id=[1],
        ),
    )


def p17_lyapunov_configs():
    """Full MLP grid for p=17 at ensemble_id=2 with the live mean-FTLE hook.
    Fresh ensemble_id so this is a clean smoke for the new analytics ingest
    (no resume from prior smoke-set checkpoints).
    """
    return get_config_grid(
        create_mlp_config,
        dict(
            width=[256],
            depth=[2, 4],
            n=[2],
            p=[17],
            frac=[0.3, 0.5, 0.7],
            weight_decay=[1.0, 0.1],
            lr=[1e-3],
            ensemble_id=[2],
        ),
    )


def create_configs():
    return p17_lyapunov_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    import sys

    if "--minimal-transformer" in sys.argv:
        distributed_train(minimal_transformer_configs())
    elif "--minimal" in sys.argv:
        distributed_train(minimal_configs())
    else:
        distributed_train(create_configs())
