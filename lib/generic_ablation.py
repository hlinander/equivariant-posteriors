import os
import torch
import plotext as plt
import itertools

from lib.train import load_or_create_state
from lib.train import do_training
from lib.ddp import ddp_setup


def generic_ablation(out_dir, create_config, values_dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    device_id = ddp_setup()

    print(f"Using device {device_id}")

    metric_ablation = {}
    combinations = itertools.product(*values_dict.values())
    kwarg_names = list(values_dict.keys())
    for values in combinations:
        kwargs = {name: val for name, val in zip(kwarg_names, values)}
        train_run = create_config(**kwargs)
        print("Create or load state...")
        state = load_or_create_state(train_run, device_id)

        print("Do training...")
        do_training(train_run, state, device_id)

        metric_ablation[values] = (
            state.train_metrics,
            state.validation_metrics,
            state.epoch,
        )

    for metric_idx in range(len(train_run.train_eval.validation_metrics)):
        plt.clf()
        plt.cld()
        plt.title(f"{state.validation_metrics[metric_idx].name()}")
        for param, (train_metrics, val_metrics, epoch) in metric_ablation.items():
            epochs = list(range(epoch))
            train_means = [train_metrics[metric_idx].mean(epoch) for epoch in epochs]
            train_means = [mean for mean in train_means if mean is not None]
            val_means = [val_metrics[metric_idx].mean(epoch) for epoch in epochs]
            val_means = [mean for mean in val_means if mean is not None]
            plt.plot(
                epochs,
                train_means,
                label=f"Train {state.validation_metrics[metric_idx].name()}",
            )
            plt.plot(
                epochs,
                val_means,
                label=f"Val {state.validation_metrics[metric_idx].name()}",
            )

        plt.show()
        plt.save_fig(out_dir / f"{state.validation_metrics[metric_idx].name()}.html")
        plt.save_fig(
            out_dir / f"{state.validation_metrics[metric_idx].name()}",
            keep_colors=True,
        )
