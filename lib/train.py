import traceback
from pathlib import Path
import torch
import logging
import time
from typing import Dict
from typing import Optional

from filelock import FileLock
from lib.metric import MetricSample, Metric
from lib.timing_metric import Timing
import lib.data_factory as data_factory
from lib.data_utils import get_sampler
import lib.model_factory as model_factory
import lib.render_duck as duck

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainEpochSpec
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import ComputeConfig

from lib.serialization import SerializeConfig
from lib.serialization import DeserializeConfig
from lib.serialization import deserialize
from lib.serialization import serialize

from lib.train_visualization import visualize_progress_batches

from lib.paths import get_lock_path
from lib.files import prepare_results
import lib.ddp as ddp


def evaluate_metrics_on_data(
    model,
    model_id,
    step,
    metrics: Dict[str, Metric],
    data_config,
    batch_size: int,
    compute_config: ComputeConfig,
    device,
):
    dataloader = create_dataloader(
        data_config, batch_size, shuffle=False, compute_config=compute_config
    )
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
            }
            output = model(batch)

            metric_sample = MetricSample(
                output=output,
                batch=batch,
                model_id=model_id,
            )
            for metric in metrics:
                values = metric.per_sample(metric_sample)

                if values.dtype == torch.double:
                    values = values.float()

                if (
                    len(values.shape) == 1
                    and values.shape[0] == batch["sample_id"].shape[0]
                ):
                    value_per_sample = values.tolist()
                else:
                    value_per_sample = None

                duck.insert_checkpoint_sample_metric(
                    metric_sample.model_id,
                    step,
                    metric.name(),
                    data_factory.get_factory().get_class(data_config).__name__,
                    batch["sample_id"].tolist(),
                    values.mean().item(),
                    value_per_sample,
                )


def validate(
    train_epoch_state: TrainEpochState,
    train_epoch_spec: TrainEpochSpec,
    train_run: TrainRun,
):
    if train_epoch_state.val_dataloader is None:
        return
    if train_epoch_state.epoch % train_run.validate_nth_epoch != 0:
        # Evaluate if this is the last epoch regardless
        if train_epoch_state.epoch != train_run.epochs:
            return
    model = train_epoch_state.model
    dataloader = train_epoch_state.val_dataloader
    device = train_epoch_spec.device_id

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
            }
            output = model(batch)
            if ddp.get_rank() == 0:
                metric_sample = MetricSample(
                    output=output,
                    batch=batch,
                    model_id=train_epoch_state.model_id,
                )
                for metric in train_epoch_state.validation_metrics:
                    values = metric.per_sample(metric_sample)

                    if values.dtype == torch.double:
                        values = values.float()

                    if (
                        len(values.shape) == 1
                        and values.shape[0] == batch["sample_id"].shape[0]
                    ):
                        value_per_sample = values.tolist()
                    else:
                        value_per_sample = None

                    duck.insert_checkpoint_sample_metric(
                        metric_sample.model_id,
                        train_epoch_state.batch,
                        metric.name(),
                        data_factory.get_factory()
                        .get_class(train_run.train_config.val_data_config)
                        .__name__,
                        batch["sample_id"].tolist(),
                        values.mean().item(),
                        value_per_sample,
                    )


def train(
    train_run: TrainRun,
    train_epoch_state: TrainEpochState,
    train_epoch_spec: TrainEpochSpec,
):
    model = train_epoch_state.model
    dataloader = train_epoch_state.train_dataloader
    loss = train_epoch_spec.loss
    optimizer = train_epoch_state.optimizer
    device = train_epoch_spec.device_id

    model.train()

    if train_run.train_config.model_pre_train_hook is not None:
        model = train_run.train_config.model_pre_train_hook(model, train_run=train_run)

    if dataloader.sampler.__class__.__name__ == "DistributedSampler":
        dataloader.sampler.set_epoch(train_epoch_state.epoch)

    visualizers = [visualize_progress_batches]
    for i, batch in enumerate(dataloader):
        batch_time = train_epoch_state.timing_metric.stop("batch")
        if batch_time is not None:
            if ddp.get_rank() == 0:
                duck.insert_train_step_metric(
                    train_epoch_state.model_id,
                    train_run.run_id,
                    "batch_time",
                    train_epoch_state.batch,
                    batch_time,
                )
        train_epoch_state.timing_metric.start("batch")
        train_epoch_state.batch += ddp.get_world_size()

        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        output = model(batch)

        loss_val = loss(output, batch)
        loss_val.backward()
        if ddp.get_rank() == 0:
            train_epoch_state.timing_metric.start("insert_duck_metric")
            duck.insert_train_step(
                train_epoch_state.model_id,
                train_run.run_id,
                train_epoch_state.batch,
                data_factory.get_factory()
                .get_class(train_run.train_config.train_data_config)
                .__name__,
                batch["sample_id"].long().tolist(),
            )
            train_epoch_state.timing_metric.stop("insert_duck_metric")

        if train_run.train_config.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_run.train_config.gradient_clipping
            )
        optimizer.step()

        if train_run.train_eval.log_gradient_norm:
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            norm = torch.cat(grads).norm()
            if ddp.get_rank() == 0:
                duck.insert_train_step_metric(
                    train_epoch_state.model_id,
                    train_run.run_id,
                    "gradient_norm",
                    train_epoch_state.batch,
                    norm.item(),
                )
        optimizer.zero_grad(set_to_none=True)

        if ddp.get_rank() > 0:
            continue

        if i == 0 and torch.cuda.is_available():
            train_epoch_state.device_memory_stats = (
                torch.cuda.memory_stats_as_nested_dict()
            )

        train_epoch_state.timing_metric.start("train_metrics")
        metric_sample = MetricSample(
            output=output,
            batch=batch,
            model_id=train_epoch_state.model_id,
        )
        for metric in train_epoch_state.train_metrics:
            value = metric(metric_sample)
            duck.insert_train_step_metric(
                metric_sample.model_id,
                train_run.run_id,
                metric.name(),
                train_epoch_state.batch,
                value,
            )
        train_epoch_state.timing_metric.stop("train_metrics")

        try:
            now = time.time()
            if now > train_epoch_state.next_visualization:
                train_epoch_state.next_visualization = time.time() + 10
                train_epoch_state.timing_metric.start("duck")
                last_sync_result = True, ""

                duck_time = train_epoch_state.timing_metric.stop("duck")
                duck.insert_train_step_metric(
                    train_epoch_state.model_id,
                    train_run.run_id,
                    "dbsync",
                    train_epoch_state.batch,
                    duck_time,
                )
                if train_run.visualize_terminal:
                    train_epoch_state.timing_metric.start("visualize")
                    visualizers[train_epoch_state.next_visualizer](
                        train_epoch_state,
                        train_run,
                        last_sync_result,
                        train_epoch_spec.device_id,
                    )
                    train_epoch_state.timing_metric.stop("visualize")
        except Exception as e:
            logging.error("Visualization failed")
            logging.error(str(e))
            print(traceback.format_exc())


def create_dataloader(
    data_config, batch_size: int, shuffle: bool, compute_config: ComputeConfig
):
    ds = data_factory.get_factory().create(data_config)
    sampler, shuffle = get_sampler(compute_config, ds, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        sampler=sampler,
        num_workers=compute_config.num_workers,
        collate_fn=ds.collate_fn if hasattr(ds, "collate_fn") else None,
        pin_memory=True,
        persistent_workers=True and compute_config.num_workers > 0,
    )
    return dataloader


def create_initial_state(train_run: TrainRun, code_path: Optional[Path], device_id):
    train_config = train_run.train_config

    train_ds = data_factory.get_factory().create(train_config.train_data_config)

    train_sampler, train_shuffle = get_sampler(
        train_run.compute_config, train_ds, shuffle=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        drop_last=False,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=train_run.compute_config.num_workers,
        collate_fn=train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None,
        pin_memory=True,
        persistent_workers=True and train_run.compute_config.num_workers > 0,
    )
    if train_config.val_data_config is not None:
        val_ds = data_factory.get_factory().create(train_config.val_data_config)
        val_sampler, val_shuffle = get_sampler(
            train_run.compute_config, val_ds, shuffle=False
        )
        assert val_shuffle is False
        val_dataloader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=train_config.batch_size,
            shuffle=val_shuffle,
            drop_last=False,
            sampler=val_sampler,
            num_workers=train_run.compute_config.num_workers,
            collate_fn=val_ds.collate_fn if hasattr(val_ds, "collate_fn") else None,
            pin_memory=True,
            persistent_workers=True and train_run.compute_config.num_workers > 0,
        )
    else:
        val_dataloader = None

    torch.manual_seed(train_config.ensemble_id)
    init_model = model_factory.get_factory().create(
        train_config.model_config,
        train_ds.__class__.data_spec(train_config.train_data_config),
    )

    if train_config.post_model_create_hook is not None:
        init_model = train_config.post_model_create_hook(
            init_model, train_run=train_run
        )

    init_model = init_model.to(torch.device(device_id))
    if train_run.compute_config.distributed:
        device_id_list = [device_id]
        init_model = torch.nn.parallel.DistributedDataParallel(
            init_model, device_ids=device_id_list, find_unused_parameters=True
        )

    opt = train_config.optimizer.optimizer(
        init_model.parameters(), **train_config.optimizer.kwargs
    )
    train_metrics = [metric() for metric in train_run.train_eval.train_metrics]
    validation_metrics = [
        metric() for metric in train_run.train_eval.validation_metrics
    ]

    return TrainEpochState(
        model_id=duck.insert_model(train_run) if ddp.get_rank() == 0 else 0,
        model=init_model,
        optimizer=opt,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        epoch=0,
        batch=0,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        next_visualization=0.0,
        next_visualizer=0,
        timing_metric=Timing(),
        code_path=code_path,
    )


def load_or_create_state(train_run: TrainRun, device_id) -> TrainEpochState:
    config = DeserializeConfig(
        train_run=train_run,
        device_id=device_id,
    )
    code_path = prepare_results("code", train_run)
    state = None
    try:
        state = deserialize(config)
        if state is not None:
            if ddp.get_rank() == 0:
                duck.insert_model_with_model_id(train_run, state.model_id)
    except Exception as e:
        print("ERROR: Failed to load checkpoint, creating a new initial state.")

    if state is None:
        state = create_initial_state(
            train_run=config.train_run,
            code_path=code_path,
            device_id=config.device_id,
        )

    return state


def do_training_unlocked(train_run: TrainRun, state: TrainEpochState, device_id):
    train_epoch_spec = TrainEpochSpec(
        loss=train_run.train_config.loss,
        device_id=device_id,
    )

    print("Serializing config...")
    serialize_config = SerializeConfig(train_run=train_run, train_epoch_state=state)

    if ddp.get_rank() == 0:
        try:
            print("Render duck")
            duck.render_duck(train_run, state)
        except duck.duckdb.duckdb.ConstraintException:
            print("Probably already synced model parameters...")

    # Start periodic export to S3 (replaces old sync() calls)
    if ddp.get_rank() == 0:
        print("Starting periodic export to S3...")
        export_thread = duck.start_periodic_export(train_run, interval_seconds=300)
        serialize(serialize_config)

    print("Run epochs...")
    while state.epoch < train_run.epochs:
        train(train_run, state, train_epoch_spec)

        state.epoch += 1
        if ddp.get_rank() == 0:
            duck.insert_checkpoint(state.model_id, state.batch, None)

        validate(state, train_epoch_spec, train_run)
        if ddp.get_rank() == 0:
            serialize(serialize_config)
        if train_run.compute_config.distributed:
            torch.distributed.barrier()

    if ddp.get_rank() > 0:
        return


def do_training(train_run: TrainRun, state: TrainEpochState, device_id):
    checkpoint_path = get_lock_path(train_run.train_config)
    lock = FileLock(f"{checkpoint_path}", 1)
    with lock:
        do_training_unlocked(train_run, state, device_id)
