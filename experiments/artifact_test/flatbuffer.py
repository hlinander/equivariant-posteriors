#!/usr/bin/env python
import torch
import numpy as np
import io

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
from lib.render_psql import setup_psql, add_artifact

from lib.flatbuffers import NPY, Component, Dimension
import flatbuffers


def create_config(mlp_dim, ensemble_id):
    loss = torch.nn.CrossEntropyLoss()

    def ce_loss(output, batch):
        return loss(output["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=MLPClassConfig(widths=[mlp_dim, mlp_dim]),
        train_data_config=DataSpiralsConfig(seed=0, N=1000),
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
        epochs=1,
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
    path = prepare_results("test", configs[0])
    setup_psql()

    builder = flatbuffers.Builder()
    xstr = builder.CreateString("x")
    ystr = builder.CreateString("y")
    mstr = builder.CreateString("m")
    array = np.array([[1, 2, 3], [4, 5, 6]])
    buffer = io.BytesIO()
    np.save(buffer, array)
    data = builder.CreateByteVector(buffer.getvalue())

    Component.Start(builder)
    Component.AddName(builder, xstr)
    Component.AddUnit(builder, mstr)
    x = Component.End(builder)

    Component.Start(builder)
    Component.AddName(builder, ystr)
    Component.AddUnit(builder, mstr)
    y = Component.End(builder)

    Dimension.StartComponentsVector(builder, 2)
    builder.PrependUOffsetTRelative(x)
    builder.PrependUOffsetTRelative(y)
    components = builder.EndVector()

    Dimension.Start(builder)
    Dimension.AddComponents(builder, components)
    d1 = Dimension.End(builder)

    NPY.StartDimsVector(builder, 1)
    builder.PrependUOffsetTRelative(d1)
    dimensions = builder.EndVector()
    NPY.Start(builder)
    NPY.AddDims(builder, dimensions)

    NPY.AddData(builder, data)
    npy = NPY.End(builder)
    builder.Finish(npy)
    bytes = builder.Output()

    with open(path / "npyspec.npyspec", "wb") as f:
        f.write(bytes)

    add_artifact(configs[0], "test.npyspec", path / "npyspec.npyspec")
