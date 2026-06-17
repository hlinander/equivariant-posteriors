import os
from lib.generic_ablation import get_config_grid

from experiments.climate.persisted_configs.train_climate_pear_temporal_atn_causal_multiseed import (
    create_config,
    ClimatesetHPConfig,
    ClimatesetDataHP,
    SwinHPClimatesetTemporalAtnCausalConfig,
    SwinHPClimatesetTemporalAtnCausal,
)
from experiments.climate.evaluation.evaluate_climate_hp import evaluate_climate
from lib.train_distributed import request_train_run
from lib.distributed_trainer import distributed_train
import lib.data_factory as data_factory
import lib.model_factory as model_factory

N_SEEDS  = int(os.environ.get("N_SEEDS",  "5"))
N_MODELS = int(os.environ.get("N_MODELS", "15"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "500"))

SEQ_LEN     = 4
WINDOW_SIZE = [4, 64]
EVAL_STEP   = 20

# Each entry is a named optimizer config to compare.
# Baseline matches lr_ablation_causal_w4_seq4_sweep.py exactly.
OPTIM_CONFIGS = [
    dict(name="baseline",            lr=2e-4, batch_size=12, weight_decay=3e-6),
    #dict(name="high_wd",             lr=2e-4, batch_size=12, weight_decay=1e-4),
    dict(name="small_batch",         lr=2e-4, batch_size=6,  weight_decay=3e-6),
    #dict(name="small_batch_high_wd", lr=2e-4, batch_size=6,  weight_decay=1e-4),
]


def create_configs():
    return get_config_grid(lambda **kw: kw, dict(
        ensemble_id=list(range(N_SEEDS)),
        climate_model_idx=list(range(N_MODELS)),
        optim_idx=list(range(len(OPTIM_CONFIGS))),
    ))


def _make_epochs(max_epoch, step=10):
    return list(range(step, max_epoch + 1, step))


def run(config):
    ensemble_id       = config["ensemble_id"]
    climate_model_idx = config["climate_model_idx"]
    optim             = OPTIM_CONFIGS[config["optim_idx"]]

    lr           = optim["lr"]
    batch_size   = optim["batch_size"]
    weight_decay = optim["weight_decay"]
    optim_name   = optim["name"]

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetTemporalAtnCausalConfig, SwinHPClimatesetTemporalAtnCausal)

    train_run = create_config(
        epoch=N_EPOCHS,
        ensemble_id=ensemble_id,
        climate_model_idx=climate_model_idx,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seq_len=SEQ_LEN,
        window_size=WINDOW_SIZE,
    )
    request_train_run(train_run)
    distributed_train([train_run])

    curried = lambda ensemble_id, **kw: create_config(
        epoch=N_EPOCHS,
        ensemble_id=ensemble_id,
        climate_model_idx=climate_model_idx,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seq_len=SEQ_LEN,
        window_size=WINDOW_SIZE,
        **kw,
    )

    model_name = train_run.train_config.train_data_config.climate_model
    epochs = _make_epochs(N_EPOCHS, step=EVAL_STEP)
    print(
        f"=== Evaluating {model_name} "
        f"(seed={ensemble_id}, optim={optim_name}, "
        f"lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, "
        f"seq_len={SEQ_LEN}, window_size={WINDOW_SIZE}), epochs={epochs} ==="
    )

    best_epoch, best_rmse = None, float("inf")
    for epoch in epochs:
        print(f"[{model_name} seed={ensemble_id} optim={optim_name}] epoch {epoch}")
        rmse = evaluate_climate(curried, epoch, variant_idx=ensemble_id)
        if rmse is not None and rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch

    if best_epoch is not None:
        print(
            f"=== BEST [{model_name} seed={ensemble_id} optim={optim_name}]: "
            f"epoch {best_epoch}, RMSE {best_rmse:.6f} ==="
        )


if __name__ == "__main__":
    configs = create_configs()
    print(f"{len(configs)} jobs:")
    for c in configs:
        optim = OPTIM_CONFIGS[c["optim_idx"]]
        print(f"  model={c['climate_model_idx']} seed={c['ensemble_id']} optim={optim['name']}")
