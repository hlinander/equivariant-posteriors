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


# def calibration_error(preds, target):
# return tm.functional.classification.calibration_error(preds, target, n_bins=15)


def create_classification_metric_dict(n_classes):
    return dict(
        accuracy=Metric(
            tm.functional.accuracy,
            metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        ),
        loss=Metric(loss, raw_output=True),
        calibration=Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
    )


def create_classification_metric_list(n_classes):
    return [
        lambda: Metric(
            tm.functional.accuracy,
            metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        ),
        lambda: Metric(loss, raw_output=True),
        lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
    ]


def create_regression_metric_list():
    return [
        # lambda: Metric(
        #     tm.functional.accuracy,
        #     metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        # ),
        lambda: Metric(loss, raw_output=True),
        # lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
    ]


def create_classification_metrics(data_visualizer, n_classes):
    return TrainEval(
        train_metrics=create_classification_metric_list(n_classes),
        validation_metrics=create_classification_metric_list(n_classes),
        # train_metrics=[
        #     lambda: Metric(
        #         tm.functional.accuracy,
        #         metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        #         # metric_kwargs=dict(num_classes=n_classes),
        #     ),
        #     lambda: Metric(loss, raw_output=True),
        #     lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
        #     # lambda: Metric(calibration_error),
        # ],
        # validation_metrics=[
        #     lambda: Metric(
        #         tm.functional.accuracy,
        #         metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        #         # metric_kwargs=dict(num_classes=n_classes),
        #     ),
        #     lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
        #     # lambda: Metric(calibration_error),
        #     lambda: Metric(loss, raw_output=True),
        # ],
        data_visualizer=data_visualizer,
    )


def create_regression_metrics(data_visualizer):
    return TrainEval(
        train_metrics=create_regression_metric_list(),
        validation_metrics=create_regression_metric_list(),
        # train_metrics=[
        #     lambda: Metric(
        #         tm.functional.accuracy,
        #         metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        #         # metric_kwargs=dict(num_classes=n_classes),
        #     ),
        #     lambda: Metric(loss, raw_output=True),
        #     lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
        #     # lambda: Metric(calibration_error),
        # ],
        # validation_metrics=[
        #     lambda: Metric(
        #         tm.functional.accuracy,
        #         metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        #         # metric_kwargs=dict(num_classes=n_classes),
        #     ),
        #     lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
        #     # lambda: Metric(calibration_error),
        #     lambda: Metric(loss, raw_output=True),
        # ],
        data_visualizer=data_visualizer,
    )
