# Weather Experiment

## Training

Configs live in `persisted_configs/`. Each config defines a `create_config(ensemble_id)` function and can be run as a sweep.

### Single run
```
uv run python run.py experiments/weather/persisted_configs/pear.py
```

### Sweep (SLURM)
```
uv run python run_slurm_sweep.py experiments/weather/persisted_configs/pear.py
```
This submits one SLURM job per ensemble member. Configs can override SLURM settings (e.g. `get_slurm_config()` in `pear_hres.py` requests fat nodes).

## Evaluation

### Via config file
Evaluates all epoch checkpoints across lead times 1-9 days:
```
CONFIG=experiments/weather/persisted_configs/pear.py EID=0 \
  uv run python run_slurm_sweep.py experiments/weather/evaluate_all_checkpoints.py
```
Set `EID` to the ensemble member id (default 0).

### Via checkpoint hash
When the config code has changed since training, use the checkpoint hash directly:
```
CONFIG=<hash> uv run python run_slurm_sweep.py experiments/weather/evaluate_all_checkpoints.py
```
This reconstructs the model from the saved `train_run.json` — no config file needed.

### Direct (without SLURM)
```
uv run python -m experiments.weather.evaluate_all_checkpoints experiments/weather/persisted_configs/pear.py
uv run python -m experiments.weather.evaluate_all_checkpoints <hash>
```

## Listing checkpoints

```
uv run python list_checkpoints.py
```
Shows all checkpoints ordered by date with project, model, data config, and number of saved epoch checkpoints.

## Configs

| Config | Model | Resolution | Notes |
|--------|-------|-----------|-------|
| `pear.py` | SwinHPPanguPad | nside 64 | 5 ensemble members |
| `pear_hres.py` | SwinHPPanguPad | nside 256 | 5 ensemble members, fat nodes |
| `pangu.py` | SwinHPPangu | nside 256 | |
| `pangu_large.py` | SwinHPPangu | nside 256 | larger model |
