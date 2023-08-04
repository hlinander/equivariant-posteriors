import torch
import torchmetrics as tm

from lib.metric import MetricFn, Metric, MetricMeanStdStore
from lib.train_dataclasses import TrainEval
from lib.lyapunov import lambda1, LyapunovMetric


def loss(preds, target):
    return torch.nn.functional.cross_entropy(preds, target, reduction="none")


def calibration_error(preds, target, n_classes):
    return tm.functional.classification.calibration_error(
        preds, target, n_bins=15, num_classes=n_classes, task="multiclass"
    )


# def lyapunov(preds, target):


# def calibration_error(preds, target):
# return tm.functional.classification.calibration_error(preds, target, n_bins=15)


# def create_classification_metrics_old(data_visualizer, n_classes):
#     return TrainEval(
#         train_metrics=[
#             lambda: Metric(
#                 tm.functional.accuracy,
#                 metric_kwargs=dict(task="multiclass", num_classes=n_classes),
#                 # metric_kwargs=dict(num_classes=n_classes),
#             ),
#             lambda: Metric(loss, raw_output=True),
#             lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
#             # lambda: Metric(calibration_error),
#         ],
#         validation_metrics=[
#             lambda: Metric(
#                 tm.functional.accuracy,
#                 metric_kwargs=dict(task="multiclass", num_classes=n_classes),
#                 # metric_kwargs=dict(num_classes=n_classes),
#             ),
#             lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
#             # lambda: Metric(calibration_error),
#             lambda: Metric(loss, raw_output=True),
#         ],
#         data_visualizer=data_visualizer,
#     )


def mean_std(metric):
    return Metric(store=MetricMeanStdStore(), metric=metric)


def create_classification_metrics(data_visualizer, n_classes):
    return TrainEval(
        train_metrics=[
            lambda: mean_std(
                MetricFn(
                    tm.functional.accuracy,
                    metric_kwargs=dict(task="multiclass", num_classes=n_classes),
                    # metric_kwargs=dict(num_classes=n_classes),
                )
            ),
            lambda: mean_std(MetricFn(loss, raw_output=True)),
            lambda: mean_std(
                MetricFn(calibration_error, metric_kwargs=dict(n_classes=n_classes))
            ),
            lambda: mean_std(LyapunovMetric()),
            # lambda: Metric(calibration_error),
        ],
        validation_metrics=[
            lambda: mean_std(
                MetricFn(
                    tm.functional.accuracy,
                    metric_kwargs=dict(task="multiclass", num_classes=n_classes),
                    # metric_kwargs=dict(num_classes=n_classes),
                )
            ),
            lambda: mean_std(
                MetricFn(calibration_error, metric_kwargs=dict(n_classes=n_classes))
            ),
            # lambda: Metric(calibration_error),
            lambda: mean_std(MetricFn(loss, raw_output=True)),
            lambda: mean_std(LyapunovMetric()),
        ],
        data_visualizer=data_visualizer,
    )
