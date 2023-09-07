from pathlib import Path
from lib.stable_hash import stable_hash


def get_or_create_checkpoint_path(train_config) -> Path:
    checkpoint = get_checkpoint_path(train_config)
    checkpoint.mkdir(exist_ok=True, parents=True)
    return checkpoint


def get_checkpoint_path(train_config) -> Path:
    config_hash = stable_hash(train_config)
    checkpoint_dir = Path("checkpoints/")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint = checkpoint_dir / f"checkpoint_{config_hash}"
    return checkpoint
