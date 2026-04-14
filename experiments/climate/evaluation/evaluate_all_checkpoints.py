import os
from lib.generic_ablation import get_config_grid

def _detect_evaluator():
    import importlib.util
    from pathlib import Path
    config_path = os.environ["CONFIG"]
    spec = importlib.util.spec_from_file_location(Path(config_path).stem, config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        train_run = mod.create_config(0, climate_model_idx=0)
    except TypeError:
        train_run = mod.create_config(0)
    is_hp = "HP" in type(train_run.train_config.train_data_config).__name__
    return "hp" if is_hp else "nohp"

_evaluator = _detect_evaluator()
if _evaluator == "nohp":
    from experiments.climate.evaluation.evaluate_climate_nohp import evaluate_climate, load_create_config
else:
    from experiments.climate.evaluation.evaluate_climate_hp import evaluate_climate, load_create_config

def create_configs():
    create_config = load_create_config(os.environ["CONFIG"])
    c = create_config(0)
    print(f"getting config grid with epochs 0 to {c.epochs} every {c.keep_nth_epoch_checkpoints} epochs, and variant_idx from 0 to {int(os.environ.get('NUM_VARIANTS', '1'))-1}")
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            epoch=list(range(0, c.epochs + 1, c.keep_nth_epoch_checkpoints)),
            variant_idx=list(range(int(os.environ.get("NUM_VARIANTS", "1")))),
        ),
    )

def run(config):
    #print(f"Evaluating variant {config['variant_idx']} at epoch {config['epoch']}...")
    create_config = load_create_config(os.environ["CONFIG"])
    print(config)
    evaluate_climate(
        create_config,
        config["epoch"],
        variant_idx=config["variant_idx"],
    )

if __name__ == "__main__":
    configs = create_configs()
    for config in configs:
        run(config())