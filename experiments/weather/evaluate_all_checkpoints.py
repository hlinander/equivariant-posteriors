import os
from experiments.weather.evaluate import evaluate_weather, load_create_config
from lib.generic_ablation import get_config_grid


def create_configs():
    create_config = load_create_config(os.environ["CONFIG"])
    c = create_config(0)
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            epoch=list(range(0, c.epochs, c.keep_nth_epoch_checkpoints)),
            lead_time_days=list(range(1, 10)),
        ),
    )


def run(config):
    create_config = load_create_config(os.environ["CONFIG"])
    evaluate_weather(create_config, config["epoch"], config["load_time_days"])
