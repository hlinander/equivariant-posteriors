import torch
import torchmetrics as tm

from lib.metric import Metric
from lib.train_dataclasses import TrainEval


def loss(preds, target):
    return torch.nn.functional.cross_entropy(preds, target, reduction="none")


def calibration_error(preds, target, n_classes):
    return tm.functional.classification.calibration_error(
        preds, target, n_bins=15, num_classes=n_classes, task="multiclass"
    )


def create_classification_metrics(data_visualizer, n_classes):
    return TrainEval(
        train_metrics=[
            lambda: Metric(
                tm.functional.accuracy,
                metric_kwargs=dict(task="multiclass", num_classes=n_classes),
            ),
            lambda: Metric(loss),
            lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
        ],
        validation_metrics=[
            lambda: Metric(
                tm.functional.accuracy,
                metric_kwargs=dict(task="multiclass", num_classes=n_classes),
            ),
            lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
            lambda: Metric(loss),
        ],
        data_visualizer=data_visualizer,
    )
