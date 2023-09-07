import torch
from dataclasses import dataclass
from typing import List
import shutil

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainRun
import lib.data_factory as data_factory
from lib.metric import Metric
from lib.paths import get_checkpoint_path
import lib.model_factory as model_factory
from lib.data_utils import get_sampler
from lib.ddp import get_rank


@dataclass
class SerializeConfig:
    train_run: TrainRun
    train_epoch_state: TrainEpochState


def serialize_metrics(metrics: List[Metric]):
    serialized_metrics = [(metric.name(), metric.serialize()) for metric in metrics]
    serialized_metrics = {name: value for name, value in serialized_metrics}
    return serialized_metrics


def serialize(config: SerializeConfig):
    if get_rank() != 0:
        # print("I am not rank 0 so not serializing...")
        return
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
        train_run=config.train_run.serialize_human(),
    )

    checkpoint, tmp_checkpoint = get_checkpoint_path(train_config)
    torch.save(data_dict, tmp_checkpoint)
    shutil.move(tmp_checkpoint, checkpoint)


@dataclass
class DeserializeConfig:
    train_run: TrainRun
    device_id: torch.device


def is_serialized(config: TrainRun):
    train_config = config.train_config
    checkpoint_path, _ = get_checkpoint_path(train_config)
    return checkpoint_path.is_file()


def deserialize(config: DeserializeConfig):
    train_config = config.train_run.train_config
    checkpoint_path, _ = get_checkpoint_path(train_config)
    if not checkpoint_path.is_file():
        return None
    else:
        print(f"{checkpoint_path}")

    data_dict = torch.load(checkpoint_path, map_location=torch.device(config.device_id))

    train_ds = data_factory.get_factory().create(train_config.train_data_config)
    val_ds = data_factory.get_factory().create(train_config.val_data_config)

    train_sampler, train_shuffle = get_sampler(config.train_run, train_ds, shuffle=True)
    val_sampler, val_shuffle = get_sampler(config.train_run, val_ds, shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        drop_last=True,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=config.train_run.compute_config.num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=True,
        sampler=val_sampler,
        num_workers=config.train_run.compute_config.num_workers,
    )

    model = model_factory.get_factory().create(
        train_config.model_config, val_ds.data_spec()
    )
    model = model.to(torch.device(config.device_id))

    if config.train_run.compute_config.distributed:
        device_id_list = [config.device_id]
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=device_id_list, find_unused_parameters=True
        )

    model.load_state_dict(data_dict["model"])

    if train_config.model_pre_train_hook is not None:
        model = train_config.model_pre_train_hook(model, config.train_run)

    # if train_config.post_model_create_hook is not None:
    # model = train_config.post_model_create_hook(model, train_run=config.train_run)

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
