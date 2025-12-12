import os
import math
import plotext as plt
import shutil
import psutil

from contextlib import redirect_stdout
import io
import torch

from lib.paths import get_or_create_checkpoint_path
from lib.stable_hash import stable_hash_small
import lib.render_duck as duck


def visualize_progress_batches(state, train_run, last_postgres_result, device):
    # plt.clt()
    plt.cld()
    plt.scatter()
    # Two columns
    plt.subplots(1, 3)

    train_metric_names = [metric.name() for metric in state.train_metrics]
    # val_metric_names = [metric.name() for metric in state.validation_metrics]

    common_metrics = list(
        set(train_metric_names)
    )  # .intersection(set(val_metric_names)))
    common_metrics = sorted(common_metrics)

    extra_metrics = ["norm_ratio"]
    all_metrics = common_metrics + extra_metrics
    n_metrics = min(5, len(all_metrics))

    # First column (many metrics in rows)
    plt.subplot(1, 1).subplots(n_metrics, 1)

    for idx in range(n_metrics):
        name = all_metrics[idx]
        batches = duck.select_train_step_metric_float(state.model_id, name)

        plt.subplot(1, 1).subplot(idx + 1, 1)
        plt.title(name)
        plt.xlabel("batches")
        use_log = name not in extra_metrics
        if len(batches) > 1:
            if use_log:
                x = [bx for bx, by in batches if by > 0]
                y = [by for bx, by in batches if by > 0]
            else:
                x = [bx for bx, by in batches if not math.isnan(by)]
                y = [by for bx, by in batches if not math.isnan(by)]
            if len(x) > 1:
                if use_log:
                    plt.yscale("log")
                plt.plot(x, y, label=f"Train {name}")

    # Second column (config)
    plt.subplot(1, 2).subplots(2, 1)
    plt.subplot(1, 2).subplot(1, 1)
    if train_run.train_eval.data_visualizer is not None:
        train_run.train_eval.data_visualizer(plt, state, device)
    plt.subplot(1, 2).subplot(2, 1).subplots(1, 3)
    plt.subplot(1, 2).subplot(2, 1).subplot(1, 1)

    if state.device_memory_stats is not None:
        plot_device_memory_stats(
            plt, filter_memory_stats(state.device_memory_stats), device
        )
    plt.subplot(1, 2).subplot(2, 1).subplot(1, 2)

    plot_host_memory_stats(plt)

    plt.subplot(1, 2).subplot(2, 1).subplot(1, 3)
    status = True
    if last_postgres_result is not None:
        status, msg = last_postgres_result
        if not status:
            plt.text(msg, 0, 0, color="red")
    background = "green" if status else "red"
    color = "white" if status else "black"
    plt.text("PSQL", 0, 1, background=background, color=color)
    # plt.text(str(last_postgres_result), 0, 0)

    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Third column
    plt.subplot(1, 3)
    plt.title("Config")
    # tc = "\n".join(text_config(asdict(train_run)))
    tc_config = text_config(train_run.serialize_human())
    tc_header = [
        f"train_run: {stable_hash_small(train_run)}",
        f"train_config: {stable_hash_small(train_run.train_config)}",
    ]
    tc = "\n".join(tc_header + tc_config)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.text(tc, 0, 1, color="black")
    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.xticks([])
    plt.yticks([])

    plt.show()

    checkpoint_path = get_or_create_checkpoint_path(train_run.train_config)
    # f = io.StringIO()
    # with redirect_stdout(f):
    #     # plt.save_fig(checkpoint_path / "batches_tmp.html")
    #     plt.save_fig(checkpoint_path / "batches_term_")
    #     plt.save_fig(checkpoint_path / "batches_term_color_", keep_colors=True)
    # # shutil.move(
    # #     checkpoint_path / "batches_tmp.html", checkpoint_path / "batches_training.html"
    # # )
    # shutil.move(
    #     checkpoint_path / "batches_term_", checkpoint_path / "batches_training.term"
    # )
    # shutil.move(
    #     checkpoint_path / "batches_term_color_",
    #     checkpoint_path / "batches_training.term_color",
    # )


def visualize_progress(state, train_run, last_postgres_result, device):
    # plt.clt()
    plt.cld()
    plt.scatter()
    # Two columns
    plt.subplots(1, 3)

    train_metric_names = [metric.name() for metric in state.train_metrics]
    val_metric_names = [metric.name() for metric in state.validation_metrics]

    common_metrics = list(set(train_metric_names).intersection(set(val_metric_names)))
    common_metrics = sorted(common_metrics)
    n_metrics = min(4, len(common_metrics))

    # First column (many metrics in rows)
    plt.subplot(1, 1).subplots(n_metrics, 1)

    for idx in range(n_metrics):
        name = common_metrics[idx]
        train_rows = duck.select_train_epoch_metric(state.model_id, name, "train")
        val_rows = duck.select_train_epoch_metric(state.model_id, name, "val")

        plt.subplot(1, 1).subplot(idx + 1, 1)
        plt.title(name)
        plt.xlabel("epoch")
        if len(train_rows) > 0:
            x = [r[0] for r in train_rows if not math.isnan(r[1]) and r[1] > 0]
            y = [r[1] for r in train_rows if not math.isnan(r[1]) and r[1] > 0]
            if len(x) > 0:
                plt.yscale("log")
                plt.plot(x, y, label=f"Train {name}")
        if len(val_rows) > 0:
            x = [r[0] for r in val_rows if not math.isnan(r[1]) and r[1] > 0]
            y = [r[1] for r in val_rows if not math.isnan(r[1]) and r[1] > 0]
            if len(x) > 0:
                plt.yscale("log")
                plt.plot(x, y, label=f"Val {name}")

    # Second column (config)
    plt.subplot(1, 2).subplots(2, 1)
    plt.subplot(1, 2).subplot(1, 1)
    if train_run.train_eval.data_visualizer is not None:
        train_run.train_eval.data_visualizer(plt, state, device)
    plt.subplot(1, 2).subplot(2, 1).subplots(1, 3)
    plt.subplot(1, 2).subplot(2, 1).subplot(1, 1)

    if state.device_memory_stats is not None:
        plot_device_memory_stats(
            plt, filter_memory_stats(state.device_memory_stats), device
        )
    plt.subplot(1, 2).subplot(2, 1).subplot(1, 2)

    plot_host_memory_stats(plt)

    plt.subplot(1, 2).subplot(2, 1).subplot(1, 3)
    status = True
    if last_postgres_result is not None:
        status, msg = last_postgres_result
        if not status:
            plt.text(msg, 0, 0, color="red")
    background = "green" if status else "red"
    color = "white" if status else "black"
    plt.text("PSQL", 0, 1, background=background, color=color)
    # plt.text(str(last_postgres_result), 0, 0)

    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Third column
    plt.subplot(1, 3)
    plt.title("Config")
    # tc = "\n".join(text_config(asdict(train_run)))
    tc_config = text_config(train_run.serialize_human())
    tc_header = [
        f"train_run: {stable_hash_small(train_run)}",
        f"train_config: {stable_hash_small(train_run.train_config)}",
    ]
    tc = "\n".join(tc_header + tc_config)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.text(tc, 0, 1, color="black")
    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.xticks([])
    plt.yticks([])

    plt.show()

    checkpoint_path = get_or_create_checkpoint_path(train_run.train_config)
    f = io.StringIO()
    with redirect_stdout(f):
        plt.save_fig(checkpoint_path / "tmp.html")
        plt.save_fig(checkpoint_path / "term_")
        plt.save_fig(checkpoint_path / "term_color_", keep_colors=True)
    shutil.move(checkpoint_path / "tmp.html", checkpoint_path / "training.html")
    shutil.move(checkpoint_path / "term_", checkpoint_path / "training.term")
    shutil.move(
        checkpoint_path / "term_color_", checkpoint_path / "training.term_color"
    )


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


def plot_device_memory_stats(plt, memory_stats: dict, device):
    if device == "cpu":
        return
    device_stats = torch.cuda.get_device_properties(device)

    def bytes_to_mb(bytes):
        return bytes / 1e6

    # keys = list(memory_stats.keys())
    keys = ["allocated_bytes"]  # list(memory_stats.keys())
    current = [bytes_to_mb(memory_stats[key]["current"]) for key in keys]
    peak = [bytes_to_mb(memory_stats[key]["peak"]) for key in keys]
    max = [bytes_to_mb(device_stats.total_memory) for key in keys]
    num_gpus = int(os.getenv("EP_NUM_GPUS", "1"))
    plt.text(f"Device ({num_gpus} gpus)", 0, 0)
    plt.multiple_bar(
        keys, [current, peak, max], label=["current", "peak", "max"], orientation="v"
    )


def plot_host_memory_stats(
    plt,
):
    def bytes_to_mb(bytes):
        return int(bytes / 1e6)

    max = [bytes_to_mb(psutil.virtual_memory().total)]
    current = [bytes_to_mb(psutil.virtual_memory().used)]
    peak = [0]
    plt.multiple_bar(
        ["RAM"], [current, peak, max], label=["current", "peak", "max"], orientation="v"
    )
    plt.title("Host")
