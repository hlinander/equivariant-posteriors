import os
import torch
import torchmetrics as tm

from lib.train import TrainConfig
from lib.train import TrainEval
from lib.train import TrainRun
from lib.metric import Metric
from lib.models.transformer import TransformerConfig
from lib.data import DataSpiralsConfig


from lib.train import load_or_create_state

import plotext as plt


def loss(preds, target):
    return torch.nn.functional.binary_cross_entropy(preds, target, reduction="none")


def main():
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"

    print(f"Using device {device_id}")

    plt.clt()

    for ensemble_idx in range(5):
        train_config = TrainConfig(
            model_config=TransformerConfig(
                embed_d=30, mlp_dim=20, n_seq=2, batch_size=500
            ),
            data_config=DataSpiralsConfig(),
            loss=torch.nn.BCELoss(),
            batch_size=500,
            epochs=300,
            ensemble_id=ensemble_idx,
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
        train_run = TrainRun(train_config=train_config, train_eval=train_eval)

        state = load_or_create_state(train_run, device_id)

        epochs = list(range(state.epoch))
        means = [state.metrics[1].mean(epoch) for epoch in epochs]
        plt.plot(epochs, means)

    plt.show()


if __name__ == "__main__":
    main()
