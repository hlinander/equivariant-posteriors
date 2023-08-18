import json
from pathlib import Path
import shutil
import lib.git as git
import lib.stable_hash as stable_hash


def create_result_path(base_path: Path, name: str, config: object):
    git_short_rev = git.get_rev_short()
    breakpoint()
    config_hash = stable_hash.stable_hash_small(config)
    path = base_path / f"{name}_{git_short_rev}_{config_hash}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_config_human(config, path):
    with open(path, "w") as f:
        f.write(json.dumps(config.serialize_human(), indent=2))


def copy_working_tree_to_destination(dest_path):
    dest_path = Path(dest_path)

    # Ensure destination path exists
    dest_path.mkdir(parents=True, exist_ok=True)

    # Iterate over all tracked files
    for file in git.GIT_REPO.head.commit.tree.traverse():
        if file.type == "blob":  # A file (not a directory)
            dest_file_path = dest_path / file.path

            # Ensure the parent directory exists
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)

            source_file = Path(git.GIT_REPO.working_tree_dir) / file.path            
            if source_file.is_file():
                # Copy the file
                shutil.copy2(source_file, dest_file_path)


def prepare_results(base_path: Path, name: str, config: object) -> Path:
    result_path = create_result_path(base_path, name, config)
    copy_working_tree_to_destination(result_path / "code")
    write_config_human(config, result_path / "config.json")
    return result_path
