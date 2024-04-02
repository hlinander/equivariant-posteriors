import json
from pathlib import Path
import shutil
from filelock import FileLock, Timeout
import lib.git as git
import lib.stable_hash as stable_hash
from lib.compute_env import env
from lib.serialize_human import serialize_human


def create_result_path(name: str, config: object):
    base_path = env().paths.artifacts
    git_short_rev = git.get_rev_short()
    config_hash = stable_hash.stable_hash_small(config)
    path = base_path / "results" / f"{name}_git_{git_short_rev}_config_{config_hash}"
    path.mkdir(parents=True, exist_ok=True)
    current_version = base_path / f"{name}_latest"
    current_version.unlink(missing_ok=True)
    current_version.symlink_to(path.relative_to(base_path))
    return path


def write_config_human(config, path):
    with open(path, "w") as f:
        f.write(json.dumps(serialize_human(config), indent=2))


def copy_tracked_tree_to_destination(dest_path):
    if not git.is_git_repo():
        print("[copy_working_tree_to_destination] Skipping, no git repo")
        return
    dest_path = Path(dest_path)

    # Ensure destination path exists
    dest_path.mkdir(parents=True, exist_ok=True)

    # Iterate over all tracked files
    for file in git.git_repo().head.commit.tree.traverse():
        if file.type == "blob":  # A file (not a directory)
            dest_file_path = dest_path / file.path

            # Ensure the parent directory exists
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)

            source_file = Path(git.git_repo().working_tree_dir) / file.path
            if source_file.is_file():
                # Copy the file
                shutil.copy2(source_file, dest_file_path)


def get_results_lock_path(name: str):
    # config_hash = stable_hash(train_run)
    lock_dir = Path("locks/")
    lock_dir.mkdir(exist_ok=True, parents=True)
    lock_path = lock_dir / f"lock_{name}_results"
    return lock_path


def prepare_results(name: str, config: object) -> Path:
    try:
        with FileLock(get_results_lock_path(name), 5):
            result_path = create_result_path(name, config)
            copy_tracked_tree_to_destination(result_path / "code")
            write_config_human(config, result_path / "config.json")
            return result_path
    except Timeout:
        print(
            "[Prepare results] This config is already locked, assuming results path is created elsewhere..."
        )
