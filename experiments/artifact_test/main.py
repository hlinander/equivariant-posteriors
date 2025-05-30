#!/usr/bin/env python
import torch

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics
from lib.data_registry import DataSpiralsConfig
from lib.datasets.spiral_visualization import visualize_spiral
from lib.models.mlp import MLPClassConfig
from lib.generic_ablation import generic_ablation

from lib.distributed_trainer import distributed_train
from lib.ddp import ddp_setup
from lib.files import prepare_results
from lib.render_duck import (
    ensure_duck,
    insert_artifact,
    sync,
    insert_model_with_model_id,
)
from lib.serialization import deserialize_model, DeserializeConfig


def create_config(mlp_dim, ensemble_id):
    loss = torch.nn.CrossEntropyLoss()

    def ce_loss(output, batch):
        return loss(output["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=MLPClassConfig(widths=[mlp_dim, mlp_dim]),
        train_data_config=DataSpiralsConfig(seed=0, N=1002),
        val_data_config=DataSpiralsConfig(seed=1, N=500),
        loss=ce_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.0001)
        ),
        batch_size=500,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_spiral, 2)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=1),
        train_config=train_config,
        train_eval=train_eval,
        epochs=4,
        save_nth_epoch=20,
        validate_nth_epoch=20,
    )
    return train_run


if __name__ == "__main__":
    configs = generic_ablation(
        create_config,
        dict(mlp_dim=[10], ensemble_id=list(range(1))),
    )
    distributed_train(configs)

    device = ddp_setup()
    dsm = deserialize_model(DeserializeConfig(train_run=configs[0], device_id=device))
    insert_model_with_model_id(configs[0], dsm.model_id)
    path = prepare_results("test", configs[0])
    # open(path / "test.npy", "wb").write(b"0" * 1024 * 1024 * 10)
    # print(path)
    # setup_psql()
    # add_artifact(configs[0], "test.npy", path / "test.bin")
    ensure_duck(configs[0])

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("agg")

    plt.plot([0, 1], [0, 1])
    plt.savefig(path / "plot.png")
    insert_artifact(dsm.model_id, "plot.png", path / "plot.png", "png")
    sync(configs[0])
