from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import plotext as plt
import shutil
import logging
from contextlib import redirect_stdout
import io
import time

from lib.metric import MetricSample
from lib.model import ModelFactory
from lib.data import DataFactory
from lib.stable_hash import stable_hash
from lib.render_dataframe import render_dataframe

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainEpochSpec
from lib.train_dataclasses import Factories
from lib.train_dataclasses import TrainRun


def train(train_epoch_state: TrainEpochState, train_epoch_spec: TrainEpochSpec):
    model = train_epoch_state.model
    dataloader = train_epoch_state.dataloader
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
        # print(f"{loss_val.detach().cpu()}")

        metric_sample = MetricSample(
            prediction=output,
            target=target,
            sample_id=sample_id,
            epoch=train_epoch_state.epoch,
        )
        for metric in train_epoch_state.metrics:
            metric(metric_sample)

    train_epoch_state.epoch += 1


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


def serialize(config: SerializeConfig):
    train_config = config.train_run.train_config
    train_epoch_state = config.train_epoch_state
    train_run = config.train_run

    if train_epoch_state.epoch % train_run.save_nth_epoch != 0:
        # Save if this is the last epoch regardless
        if train_epoch_state.epoch != train_run.epochs:
            return
    serialized_metrics = [
        (metric.name(), metric.serialize()) for metric in train_epoch_state.metrics
    ]
    serialized_metrics = {name: value for name, value in serialized_metrics}
    data_dict = dict(
        model=train_epoch_state.model.state_dict(),
        optimizer=train_epoch_state.optimizer.state_dict(),
        epoch=train_epoch_state.epoch,
        metrics=serialized_metrics,
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

    ds = config.factories.data_factory.create(train_config.data_config)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = config.factories.model_factory.create(
        train_config.model_config,
        train_config.data_config,
    )
    model = model.to(torch.device(config.device_id))
    model.load_state_dict(data_dict["model"])

    optimizer = train_config.optimizer.optimizer(
        model.parameters(), **train_config.optimizer.kwargs
    )
    optimizer.load_state_dict(data_dict["optimizer"])

    metrics = []
    for metric in config.train_run.train_eval.metrics:
        metric_instance = metric()
        metric_instance.deserialize(data_dict["metrics"][metric_instance.name()])
        metrics.append(metric_instance)

    epoch = data_dict["epoch"]

    return TrainEpochState(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=metrics,
        dataloader=dataloader,
    )


def create_initial_state(
    models: ModelFactory, datasets: DataFactory, train_run: TrainRun, device_id
):
    train_config = train_run.train_config
    ds = datasets.create(train_config.data_config)
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=train_config.batch_size, shuffle=True, drop_last=True
    )

    model = models.create(train_config.model_config, train_config.data_config)
    model = model.to(torch.device(device_id))
    opt = torch.optim.Adam(model.parameters())
    metrics = [metric() for metric in train_run.train_eval.metrics]

    return TrainEpochState(
        model=model, optimizer=opt, metrics=metrics, epoch=0, dataloader=dataloader
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
        serialize(serialize_config)
        try:
            now = time.time()
            if now > next_visualization:
                next_visualization = now + 1
                visualize_progress(state, train_run)
        except Exception as e:
            logging.error("Visualization failed")
            logging.error(str(e))

    df = render_dataframe(train_run, state)
    df.to_pickle(path=f"{get_checkpoint_path(train_run.train_config)[0]}.df.pickle")


def visualize_progress(state, train_run):
    plt.clt()
    plt.cld()
    epochs = list(range(state.epoch))
    n_metrics = min(4, len(state.metrics))
    # Two columns
    plt.subplots(1, 2)

    # First column (many metrics in rows)
    plt.subplot(1, 1).subplots(n_metrics, 1)
    for idx in range(n_metrics):
        means = [state.metrics[idx].mean(epoch) for epoch in epochs]
        plt.subplot(1, 1).subplot(idx + 1, 1)
        plt.title(state.metrics[idx].name())
        plt.plot(epochs, means, label=f"{state.metrics[idx].name()}")

    # Second column (config)
    plt.subplot(1, 2)
    plt.title("Config")
    tc = "\n".join(text_config(asdict(train_run)))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.text(tc, 0, 1)
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
