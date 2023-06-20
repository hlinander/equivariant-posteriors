from pathlib import Path
from lib.stable_hash import stable_hash


def get_checkpoint_path(train_config) -> (Path, Path):
    config_hash = stable_hash(train_config)
    checkpoint_dir = Path("checkpoints/")
    checkpoint_dir.mkdir(exist_ok=True)
    tmp_checkpoint = checkpoint_dir / f"_checkpoint_{config_hash}.pt"
    checkpoint = checkpoint_dir / f"checkpoint_{config_hash}.pt"
    return checkpoint, tmp_checkpoint
