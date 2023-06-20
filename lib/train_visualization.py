import plotext as plt
import shutil
from dataclasses import asdict

from contextlib import redirect_stdout
import io
import torch

from lib.paths import get_checkpoint_path


def visualize_progress(state, train_run, device):
    # plt.clt()
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
        train_run.train_eval.data_visualizer(plt, state, device)
    plt.subplot(1, 2).subplot(2, 1)
    plt.title("Config")
    tc = "\n".join(text_config(asdict(train_run)))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.text(tc, 0, 1, color="black")

    # Third column
    plt.subplot(1, 3)
    if state.device_memory_stats is not None:
        plot_device_memory_stats(
            plt, filter_memory_stats(state.device_memory_stats), device
        )

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


def plot_device_memory_stats(plt, memory_stats: dict, device):
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