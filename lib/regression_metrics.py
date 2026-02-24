import torch
import torchmetrics as tm

from lib.metric import Metric
from lib.train_dataclasses import TrainEval


def create_regression_metric_list(loss):
    def non_reducing_loss(output, batch):
        return loss(output["predictions"], batch["target"], reduction="none")

    return [
        lambda: Metric(non_reducing_loss),
    ]


def create_regression_metrics(loss, data_visualizer):
    return TrainEval(
        train_metrics=create_regression_metric_list(loss),
        validation_metrics=create_regression_metric_list(loss),
        data_visualizer=data_visualizer,
    )
