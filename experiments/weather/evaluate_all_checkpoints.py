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


def _get_min_epoch():
    return int(os.environ.get("MIN_EPOCH", "0"))


def _get_max_epoch():
    s = os.environ.get("MAX_EPOCH")
    return int(s) if s else None


def _get_lead_range():
    """Inclusive (MIN_LEAD, MAX_LEAD) defaulting to (1, 9)."""
    return (
        int(os.environ.get("MIN_LEAD", "1")),
        int(os.environ.get("MAX_LEAD", "9")),
    )


def _get_eids():
    """Comma-separated EIDS overrides single EID; defaults to [EID or 0]."""
    eids_env = os.environ.get("EIDS")
    if eids_env:
        return [int(x) for x in eids_env.split(",") if x.strip()]
    return [int(os.environ.get("EID", "0"))]


def _filter_epochs(epochs):
    min_e = _get_min_epoch()
    max_e = _get_max_epoch()
    return [e for e in epochs if e >= min_e and (max_e is None or e <= max_e)]


def create_configs():
    source = _get_source()
    if _is_config_file(source):
        create_config = load_create_config(source)
        c = create_config(0)
        epochs = list(range(0, c.epochs, c.keep_nth_epoch_checkpoints))
    else:
        epochs = list_checkpoint_epochs(source)
    epochs = _filter_epochs(epochs)
    min_lead, max_lead = _get_lead_range()
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            epoch=epochs,
            lead_time_days=list(range(min_lead, max_lead + 1)),
            ensemble_id=_get_eids(),
        ),
    )


def run(config):
    source = _get_source()
    if _is_config_file(source):
        create_config = load_create_config(source)
        evaluate_weather(
            create_config,
            config["epoch"],
            config["lead_time_days"],
            ensemble_id=config["ensemble_id"],
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
    epochs = _filter_epochs(epochs)

    min_lead, max_lead = _get_lead_range()
    for epoch in epochs:
        for lead_time_days in range(min_lead, max_lead + 1):
            if _is_config_file(args.source):
                evaluate_weather(
                    create_config, epoch, lead_time_days,
                    ensemble_id=args.ensemble_id,
                )
            else:
                evaluate_weather_from_checkpoint(
                    args.source, epoch, lead_time_days,
                )
