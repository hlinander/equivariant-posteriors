#!/usr/bin/env python
import os
import torch
import torchmetrics as tm
import plotext as plt

from lib.train import TrainConfig
from lib.train import TrainEval
from lib.train import TrainRun
from lib.metric import Metric
from lib.models.transformer import TransformerConfig
from lib.data import DataSpiralsConfig


from lib.train import load_or_create_state
from lib.train import do_training


def loss(preds, target):
    return torch.nn.functional.binary_cross_entropy(preds, target, reduction="none")


def main():
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"

    print(f"Using device {device_id}")

    metrics = {}
    for embed_d in [1, 5, 10, 20, 50, 100]:
        train_config = TrainConfig(
            model_config=TransformerConfig(
                embed_d=embed_d, mlp_dim=10, n_seq=2, batch_size=500
            ),
            data_config=DataSpiralsConfig(),
            loss=torch.nn.BCELoss(),
            batch_size=500,
            ensemble_id=0,
        )
        train_eval = TrainEval(
            metrics=[
                lambda: Metric(
                    tm.functional.accuracy,
                    metric_kwargs=dict(task="binary", multidim_average="samplewise"),
                ),
                lambda: Metric(loss),
            ],
        )
        train_run = TrainRun(
            train_config=train_config,
            train_eval=train_eval,
            epochs=300,
        )

        state = load_or_create_state(train_run, device_id)

        do_training(train_run, state, device_id)

        metrics[embed_d] = (state.metrics, state.epoch)

    for embed_d, (metrics, epoch) in metrics.items():
        epochs = list(range(epoch))
        means = [metrics[0].mean(epoch) for epoch in epochs]
        plt.plot(epochs, means, label=f"{embed_d}")

    plt.title("embed d")
    plt.show()
    plt.save_fig("./embed_d.html")


if __name__ == "__main__":
    main()
