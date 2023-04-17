import pandas
import json

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import Factories


def dict_to_normalized_json(input_dict):
    return json.loads(pandas.json_normalize(input_dict).to_json(orient="records"))[0]


def render_dataframe(train_run: TrainRun, train_epoch_state: TrainEpochState):
    factories = Factories()
    config_cols = [
        value
        for key, value in dict_to_normalized_json(
            train_run.serialize_human(factories)
        ).items()
    ]
    config_headers = [
        key
        for key, value in dict_to_normalized_json(
            train_run.serialize_human(factories)
        ).items()
    ]
    metric_headers = [metric.name() for metric in train_epoch_state.metrics]
    headers = ["epoch"] + config_headers + metric_headers
    rows = []
    for epoch in range(train_epoch_state.epoch):
        metric_values = [metric.mean(epoch) for metric in train_epoch_state.metrics]
        rows.append([epoch] + config_cols + metric_values)

    return pandas.DataFrame(columns=headers, data=rows)
