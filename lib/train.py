import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import torch
import plotext as plt
import shutil
import logging
from contextlib import redirect_stdout
import io
import time

from lib.metric import MetricSample
from lib.metric import Metric
from lib.model import ModelFactory
from lib.data import DataFactory
from lib.stable_hash import stable_hash
from lib.render_dataframe import render_dataframe
from lib.render_psql import render_psql

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainEpochSpec
from lib.train_dataclasses import Factories
from lib.train_dataclasses import TrainRun


def validate(
    train_epoch_state: TrainEpochState,
    train_epoch_spec: TrainEpochSpec,
    train_run: TrainRun,
):
    if train_epoch_state.epoch % train_run.validate_nth_epoch != 0:
        # Evaluate if this is the last epoch regardless
        if train_epoch_state.epoch != train_run.epochs:
            return
    model = train_epoch_state.model
    dataloader = train_epoch_state.val_dataloader
    device = train_epoch_spec.device_id

    model.eval()

    with torch.no_grad():
        for i, (input, target, sample_id) in enumerate(dataloader):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(input)

            metric_sample = MetricSample(
                output=output,
                prediction=model.output_to_value(output),
                target=target,
                sample_id=sample_id,
                epoch=train_epoch_state.epoch,
            )
            for metric in train_epoch_state.validation_metrics:
                metric(metric_sample)


def train(train_epoch_state: TrainEpochState, train_epoch_spec: TrainEpochSpec):
    model = train_epoch_state.model
    dataloader = train_epoch_state.train_dataloader
    loss = train_epoch_spec.loss
    optimizer = train_epoch_state.optimizer
    device = train_epoch_spec.device_id

    model.train()

    for i, (input, target, sample_id) in enumerate(dataloader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(input)

        loss_val = loss(output, target)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        if i == 0:
            train_epoch_state.memory_stats = torch.cuda.memory_stats_as_nested_dict()

        metric_sample = MetricSample(
            output=output,
            prediction=model.output_to_value(output),
            target=target,
            sample_id=sample_id,
            epoch=train_epoch_state.epoch,
        )
        for metric in train_epoch_state.train_metrics:
            metric(metric_sample)


def get_checkpoint_path(train_config) -> (Path, Path):
    config_hash = stable_hash(train_config)
    checkpoint_dir = Path("checkpoints/")
    checkpoint_dir.mkdir(exist_ok=True)
    tmp_checkpoint = checkpoint_dir / f"_checkpoint_{config_hash}.pt"
    checkpoint = checkpoint_dir / f"checkpoint_{config_hash}.pt"
    return checkpoint, tmp_checkpoint


@dataclass
class SerializeConfig:
    factories: Factories
    train_run: TrainRun
    train_epoch_state: TrainEpochState


def serialize_metrics(metrics: List[Metric]):
    serialized_metrics = [(metric.name(), metric.serialize()) for metric in metrics]
    serialized_metrics = {name: value for name, value in serialized_metrics}
    return serialized_metrics


def serialize(config: SerializeConfig):
    train_config = config.train_run.train_config
    train_epoch_state = config.train_epoch_state
    train_run = config.train_run

    if train_epoch_state.epoch % train_run.save_nth_epoch != 0:
        # Save if this is the last epoch regardless
        if train_epoch_state.epoch != train_run.epochs:
            return
    data_dict = dict(
        model=train_epoch_state.model.state_dict(),
        optimizer=train_epoch_state.optimizer.state_dict(),
        epoch=train_epoch_state.epoch,
        train_metrics=serialize_metrics(train_epoch_state.train_metrics),
        validation_metrics=serialize_metrics(train_epoch_state.validation_metrics),
        train_run=config.train_run.serialize_human(config.factories),
    )

    checkpoint, tmp_checkpoint = get_checkpoint_path(train_config)
    torch.save(data_dict, tmp_checkpoint)
    shutil.move(tmp_checkpoint, checkpoint)


@dataclass
class DeserializeConfig:
    factories: Factories
    train_run: TrainRun
    device_id: torch.device


def deserialize(config: DeserializeConfig):
    train_config = config.train_run.train_config
    checkpoint_path, _ = get_checkpoint_path(train_config)
    if not checkpoint_path.is_file():
        return None
    else:
        print(f"{checkpoint_path}")

    data_dict = torch.load(checkpoint_path)

    train_ds = config.factories.data_factory.create(train_config.train_data_config)
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_ds = config.factories.data_factory.create(train_config.val_data_config)
    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    model = config.factories.model_factory.create(
        train_config.model_config, val_ds.data_spec()
    )
    model = model.to(torch.device(config.device_id))
    model.load_state_dict(data_dict["model"])

    optimizer = train_config.optimizer.optimizer(
        model.parameters(), **train_config.optimizer.kwargs
    )
    optimizer.load_state_dict(data_dict["optimizer"])

    train_metrics = []
    for metric in config.train_run.train_eval.train_metrics:
        metric_instance = metric()
        metric_instance.deserialize(data_dict["train_metrics"][metric_instance.name()])
        train_metrics.append(metric_instance)

    validation_metrics = []
    for metric in config.train_run.train_eval.validation_metrics:
        metric_instance = metric()
        metric_instance.deserialize(
            data_dict["validation_metrics"][metric_instance.name()]
        )
        validation_metrics.append(metric_instance)

    epoch = data_dict["epoch"]

    return TrainEpochState(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )


def create_initial_state(
    models: ModelFactory, datasets: DataFactory, train_run: TrainRun, device_id
):
    train_config = train_run.train_config
    train_ds = datasets.create(train_config.train_data_config)
    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_config.batch_size, shuffle=True, drop_last=True
    )
    val_ds = datasets.create(train_config.val_data_config)
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=train_config.batch_size, shuffle=False, drop_last=True
    )

    torch.manual_seed(train_config.ensemble_id)
    model = models.create(train_config.model_config, train_ds.data_spec())
    model = model.to(torch.device(device_id))
    opt = torch.optim.Adam(model.parameters(), **train_config.optimizer.kwargs)
    train_metrics = [metric() for metric in train_run.train_eval.train_metrics]
    validation_metrics = [
        metric() for metric in train_run.train_eval.validation_metrics
    ]

    return TrainEpochState(
        model=model,
        optimizer=opt,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        epoch=0,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )


def load_or_create_state(train_run: TrainRun, device_id):
    models = ModelFactory()
    datasets = DataFactory()
    factories = Factories(model_factory=models, data_factory=datasets)

    config = DeserializeConfig(
        factories=factories,
        train_run=train_run,
        device_id=device_id,
    )

    state = deserialize(config)

    if state is None:
        state = create_initial_state(
            models=models,
            datasets=datasets,
            train_run=config.train_run,
            device_id=config.device_id,
        )

    return state


def do_training(train_run: TrainRun, state: TrainEpochState, device_id):
    next_visualization = time.time()
    train_epoch_spec = TrainEpochSpec(
        loss=train_run.train_config.loss,
        device_id=device_id,
    )
    factories = Factories(model_factory=ModelFactory(), data_factory=DataFactory())

    serialize_config = SerializeConfig(
        factories=factories, train_run=train_run, train_epoch_state=state
    )

    while state.epoch <= train_run.epochs:
        train(state, train_epoch_spec)
        validate(state, train_epoch_spec, train_run)
        state.epoch += 1
        serialize(serialize_config)
        try:
            now = time.time()
            if now > next_visualization:
                next_visualization = now + 1
                visualize_progress(state, train_run, device_id)
        except Exception as e:
            logging.error("Visualization failed")
            logging.error(str(e))
            print(traceback.format_exc())

    df = render_dataframe(train_run, state)
    render_psql(train_run, state)
    df.to_pickle(path=f"{get_checkpoint_path(train_run.train_config)[0]}.df.pickle")


def visualize_progress(state, train_run, device):
    plt.clt()
    plt.cld()
    plt.scatter()
    epochs = list(range(state.epoch))
    # Two columns
    plt.subplots(1, 3)

    train_metric_names = [metric.name() for metric in state.train_metrics]
    val_metric_names = [metric.name() for metric in state.validation_metrics]

    common_metrics = list(set(train_metric_names).intersection(set(val_metric_names)))
    common_metrics = sorted(common_metrics)
    n_metrics = min(4, len(common_metrics))

    train_indices = [train_metric_names.index(name) for name in common_metrics]
    val_indices = [val_metric_names.index(name) for name in common_metrics]

    # First column (many metrics in rows)
    plt.subplot(1, 1).subplots(n_metrics, 1)

    for idx in range(n_metrics):
        train_metric = state.train_metrics[train_indices[idx]]
        val_metric = state.validation_metrics[val_indices[idx]]

        train_means = [(epoch, train_metric.mean(epoch)) for epoch in epochs]
        train_means = [
            (epoch, mean) for (epoch, mean) in train_means if mean is not None
        ]
        val_means = [(epoch, val_metric.mean(epoch)) for epoch in epochs]
        val_means = [(epoch, mean) for (epoch, mean) in val_means if mean is not None]
        plt.subplot(1, 1).subplot(idx + 1, 1)
        plt.title(common_metrics[idx])
        if len(train_means) > 0:
            x = [epoch for (epoch, mean) in train_means]
            y = [mean for (epoch, mean) in train_means]
            plt.plot(x, y, label=f"Train {common_metrics[idx]}")
        if len(val_means) > 0:
            x = [epoch for (epoch, mean) in val_means]
            y = [mean for (epoch, mean) in val_means]
            plt.plot(x, y, label=f"Val {common_metrics[idx]}")

    # Second column (config)
    plt.subplot(1, 2).subplots(2, 1)
    plt.subplot(1, 2).subplot(1, 1)
    if train_run.train_eval.data_visualizer is not None:
        train_run.train_eval.data_visualizer(plt, state)
    plt.subplot(1, 2).subplot(2, 1)
    plt.title("Config")
    tc = "\n".join(text_config(asdict(train_run)))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.text(tc, 0, 1, color="black")

    # Third column
    plt.subplot(1, 3)
    plot_memory_stats(plt, filter_memory_stats(state.memory_stats), device)
    # tc = "\n".join(text_config(filter_memory_stats(state.memory_stats)))
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.text(tc, 0, 1, color="black")

    plt.show()

    checkpoint_path, _ = get_checkpoint_path(train_run.train_config)
    f = io.StringIO()
    with redirect_stdout(f):
        plt.save_fig(f"{checkpoint_path}.tmp.html")
        plt.save_fig(f"{checkpoint_path}.term_")
        plt.save_fig(f"{checkpoint_path}.term_color_", keep_colors=True)
    shutil.move(f"{checkpoint_path}.tmp.html", f"{checkpoint_path}.html")
    shutil.move(f"{checkpoint_path}.term_", f"{checkpoint_path}.term")
    shutil.move(f"{checkpoint_path}.term_color_", f"{checkpoint_path}.term_color")


def text_config(config, level=0, y=0):
    text = []
    for key, value in config.items():
        if isinstance(value, dict):
            text.append(f"{'  '*level}{key}:")
            text = text + text_config(value, level + 1)
        else:
            text.append(f"{'  '*level}{key}: {value}")
    return text


def filter_memory_stats(memory_stats: dict):
    return {
        key: value["all"]
        for key, value in memory_stats.items()
        if isinstance(value, dict)
        and "all" in value
        and key in ["allocated_bytes", "reserved_bytes", "active_bytes"]
    }


def plot_memory_stats(plt, memory_stats: dict, device):
    device_stats = torch.cuda.get_device_properties(device)

    def bytes_to_mb(bytes):
        return bytes / 1e6

    keys = list(memory_stats.keys())
    current = [bytes_to_mb(memory_stats[key]["current"]) for key in keys]
    peak = [bytes_to_mb(memory_stats[key]["peak"]) for key in keys]
    max = [bytes_to_mb(device_stats.total_memory) for key in keys]
    plt.stacked_bar(
        keys, [current, peak, max], label=["current", "peak", "max"], orientation="v"
    )
