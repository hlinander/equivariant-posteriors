import argparse
import os
from experiments.weather.evaluate import (
    evaluate_weather,
    evaluate_weather_from_checkpoint,
    load_create_config,
)
from lib.serialization import list_checkpoint_epochs
from lib.generic_ablation import get_config_grid


def _is_config_file(s):
    return s.endswith(".py")


def _get_source():
    return os.environ["CONFIG"]


def create_configs():
    source = _get_source()
    if _is_config_file(source):
        create_config = load_create_config(source)
        c = create_config(0)
        epochs = list(range(0, c.epochs, c.keep_nth_epoch_checkpoints))
    else:
        epochs = list_checkpoint_epochs(source)
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            epoch=epochs,
            lead_time_days=list(range(1, 10)),
        ),
    )


def run(config):
    source = _get_source()
    if _is_config_file(source):
        create_config = load_create_config(source)
        ensemble_id = int(os.environ.get("EID", default="0"))
        evaluate_weather(
            create_config,
            config["epoch"],
            config["lead_time_days"],
            ensemble_id=ensemble_id,
        )
    else:
        evaluate_weather_from_checkpoint(
            source,
            config["epoch"],
            config["lead_time_days"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate weather model across all epoch checkpoints and lead times."
    )
    parser.add_argument(
        "source",
        help="Config .py file path, or a checkpoint hash hex string.",
    )
    parser.add_argument("--ensemble-id", type=int, default=0)
    args = parser.parse_args()

    if _is_config_file(args.source):
        create_config = load_create_config(args.source)
        c = create_config(0)
        epochs = list(range(0, c.epochs, c.keep_nth_epoch_checkpoints))
    else:
        epochs = list_checkpoint_epochs(args.source)
        if not epochs:
            print(f"No epoch checkpoints found for {args.source}")
            exit(1)

    for epoch in epochs:
        for lead_time_days in range(1, 10):
            if _is_config_file(args.source):
                evaluate_weather(
                    create_config, epoch, lead_time_days,
                    ensemble_id=args.ensemble_id,
                )
            else:
                evaluate_weather_from_checkpoint(
                    args.source, epoch, lead_time_days,
                )
