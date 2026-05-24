#!/usr/bin/env python
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.classification_metrics import create_classification_metrics
from lib.models.grok_mlp import GrokMLPConfig
from lib.models.transformer import TransformerConfig
from lib.datasets.finite_field_det import DataFiniteFieldDetConfig
from lib.generic_ablation import get_config_grid
from lib.distributed_trainer import distributed_train


def _make_train_run(model_config, n, p, frac, weight_decay, lr, ensemble_id, seq=False):
    loss = torch.nn.CrossEntropyLoss()

    def ce_loss(output, batch):
        return loss(output["logits"], batch["target"])

    train_eval = create_classification_metrics(None, p)
    train_eval.log_gradient_norm = True
    train_eval.log_parameter_norm = True
    train_eval.diagnostics_interval = 10

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
        loss=ce_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(lr=lr, weight_decay=weight_decay),
        ),
        batch_size=min(train_data.n_samples, 10000),
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
        validate_nth_epoch=50,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=5000,
        visualize_interval_s=5,
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


def create_configs():
    return p13_sweep_configs()


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
