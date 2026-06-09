[![CI](https://github.com/hlinander/equivariant-posteriors/actions/workflows/main.yml/badge.svg)](https://github.com/hlinander/equivariant-posteriors/actions/workflows/main.yml)

# Equivariant posteriors
Train and evaluate PyTorch models with reproducibility in mind.

- Computational environment reproducible through Nix flake.
- Python based configuration in terms of dataclasses.
- Convenient metric functionality with focus on saving as much as possible for future inspection.
- Simple TUI for easy progress inspection.
- DuckDB-based storage with S3 ingestion pipeline for scalable run tracking and visualization.


## Quick Start with uv

Install [uv](https://github.com/astral-sh/uv) and run experiments:

```bash
uv run run.py experiments/mnist/dense.py            # auto-detect device
uv run run.py --mode cuda experiments/mnist/dense.py
uv run run.py --mode cpu experiments/mnist/dense.py
```

On first run, `env.py` is auto-created with defaults. All data is stored under `.local/` (add to `.gitignore`).

### Secrets (SOPS + age)

The analytics pipeline (S3 staging credentials) reads its secrets from a
SOPS-encrypted file at the repo root. Only leaf values are ciphertext —
diffs stay reviewable.

```
.sops.yaml             # recipient list (age public keys)
secrets.example.yaml   # plaintext template (committed)
secrets.enc.yaml       # encrypted, committed
secrets.yaml           # plaintext, gitignored — only exists transiently
lib/secrets.py         # tiny `sops -d` loader, cached per process
```

Bootstrap (once per machine):

```bash
# 1. install sops + age
#    macOS:  brew install sops age
#    linux:  download static binaries from each project's GitHub releases
mkdir -p ~/.config/sops/age
age-keygen -o ~/.config/sops/age/keys.txt
grep "public key" ~/.config/sops/age/keys.txt        # copy this line

# 2. paste the age1... public key into .sops.yaml under `age:`
# 3. encrypt your secrets
cp secrets.example.yaml secrets.yaml
$EDITOR secrets.yaml                                 # paste real values
sops -e secrets.yaml > secrets.enc.yaml
rm secrets.yaml
```

Day-to-day: `sops secrets.enc.yaml` decrypts into `$EDITOR` and re-encrypts on
save. Anything that imports `env.py` will pick up the new values on next start.

Adding a teammate: append their age public key to `.sops.yaml`, then
`sops updatekeys secrets.enc.yaml` to re-wrap the data key for the new
recipient.

### SLURM

Submit a single experiment to SLURM:

```bash
python run_slurm.py experiments/weather/train.py
python run_slurm.py --dry-run experiments/weather/train.py  # print script only
```

Submit a sweep (array job) to SLURM:

```bash
python run_slurm_sweep.py experiments/weather/sweep.py
python run_slurm_sweep.py --dry-run experiments/weather/sweep.py
python run_slurm_sweep.py --max-concurrent 4 experiments/weather/sweep.py
python run_slurm_sweep.py --run-local experiments/weather/sweep.py  # test locally
```

A sweep file defines `create_configs()` returning a list of callables and `run(config)`:

```python
def create_configs():
    return [lambda lr=lr: {"lr": lr} for lr in [1e-2, 1e-3, 1e-4]]

def run(config):
    print(f"Training with {config}")
```

SLURM parameters (time, GPUs, partition, etc.) are configured via `get_slurm_config()` in `env.py`. Sweep files can override this by defining their own `get_slurm_config()`.

### Local Ingestion

After training, ingest metrics into the central database for querying:

```bash
uv run python ingestion/ingest.py
```

Query results with DuckDB:

```python
import duckdb
conn = duckdb.connect(".local/analytics.db")
conn.execute("SELECT * FROM train_step_metric_float WHERE name = 'loss'").fetchdf()
```

## Nix
This project uses [Nix](https://nixos.org/) for reproducible compute environmens.

### Binary cache from [Cachix](https://www.cachix.org/)
We provide cached builds of dependencies for a CUDA enabled system through Cachix. See instructions at
[https://app.cachix.org/cache/equivariant-posteriors#pull](https://app.cachix.org/cache/equivariant-posteriors#pull).

It is probably also a good idea to enable the CUDA maintainers cache:
https://app.cachix.org/cache/cuda-maintainers#pull

### Install
Install nix to your home-folder or system wide.
```
  https://nixos.org/download.html#download-nix
```
Enable [flakes](https://zero-to-nix.com/concepts/flakes) by one of

 1. Nix in other distribution: Create (or add to) `~/.config/nix/nix.conf` (or `/etc/nix/nix.conf`)
    ```
      experimental-features = nix-command flakes
    ```
 2. NixOS: Add to `/etc/nixos/configuration.nix`
    ```
      nix = {
        package = pkgs.nixFlakes;
        extraOptions = ''
          experimental-features = nix-command flakes
        '';
      };
    ```


### Development 
Start a development shell
```
  nix develop
```

### Test project
To build and run tests
```
  nix build
```

### Build singularity image
Build a singularity image with CUDA support containing the project
```
  nix build .#sing
```
