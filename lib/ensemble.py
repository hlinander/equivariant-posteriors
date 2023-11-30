import time
from typing import Callable, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
from lib.train_dataclasses import TrainRun
from lib.train import load_or_create_state
from lib.train import do_training
from lib.train_distributed import request_train_run, lock_requested_hash
from lib.serialization import (
    get_checkpoint_path,
    deserialize_model,
    DeserializeConfig,
    is_serialized,
)
from lib.model_factory import get_factory
from lib.stable_hash import stable_hash


@dataclass
class EnsembleConfig:
    members: list[TrainRun]

    def serialize_human(self):
        return [member_config.serialize_human() for member_config in self.members]


@dataclass
class Ensemble:
    member_configs: list[TrainRun]
    members: list[torch.nn.Module]
    n_members: int


def create_ensemble_config(
    create_member_config: Callable[[int], TrainRun], n_members: int
):
    return EnsembleConfig(
        members=[create_member_config(member_id) for member_id in range(n_members)]
    )


def get_ensemble_checkpoint_files(ensemble: Ensemble) -> List[Tuple[TrainRun, Path]]:
    checkpoint_files = []
    for member_config in ensemble.member_configs:
        checkpoint_path = get_checkpoint_path(member_config.train_config)
        checkpoint_files += [
            (member_config, path.absolute()) for path in checkpoint_path.glob("*")
        ]
    return checkpoint_files


def symlink_checkpoint_files(ensemble, target_path: Path):
    output_path = target_path / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    model_factory = get_factory()
    for config, file in get_ensemble_checkpoint_files(ensemble):
        model_name = model_factory.get_class(config.train_config.model_config).__name__
        link_path = output_path / model_name / file.name
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(file)


def train_member(ensemble_config: EnsembleConfig, member_idx: int, device_id):
    if member_idx < 0 or member_idx > len(ensemble_config.members):
        print(
            f"Member idx {member_idx} is not in index set for ensemble {ensemble_config}"
        )
        return
    member_config = ensemble_config.members[member_idx]
    state = load_or_create_state(member_config, device_id)
    # if is_serialized(member_config):
    #     if state.
    #     print(f"Member {member_idx} already available")
    #     return
    do_training(member_config, state, device_id)


def create_ensemble(ensemble_config: EnsembleConfig, device_id):
    members = []
    all_member_hashes = set(
        [stable_hash(member_config) for member_config in ensemble_config.members]
    )
    train_config_hash_dict = {
        stable_hash(member_config): stable_hash(member_config.train_config)
        for member_config in ensemble_config.members
    }
    trained_members = set()
    print("Will try to load ensemble checkpoints from:")
    for member_config in ensemble_config.members:
        if not is_serialized(member_config):
            request_train_run(member_config)
        print(get_checkpoint_path(member_config.train_config).as_posix())
    print("Loading or training ensemble...")
    while len(trained_members) < len(ensemble_config.members):
        for member_config in ensemble_config.members:
            hash = stable_hash(member_config)
            print(f"Checking if {hash} has already been trained...")
            if stable_hash(member_config) in trained_members:
                print(f"{hash} has already been trained. Continuing.")
                continue

            print(f"Trying to aquire distributed lock for train run {hash}")
            distributed_train_run_lock = lock_requested_hash(hash)
            if distributed_train_run_lock is None:
                print(f"Could not aquire distributed lock for {hash}. Continuing.")
                continue
            print(f"Aquired lock for {hash}.")

            try:
                print(f"Deserializing {hash}")
                deserialized_model = deserialize_model(
                    DeserializeConfig(member_config, device_id)
                )
                if (
                    deserialized_model is None
                    or deserialized_model.epoch < member_config.epochs
                ):
                    print(
                        f"Could not deserialize {hash} or epochs are not enough, I will continue training myself."
                    )
                    state = load_or_create_state(member_config, device_id)
                    # print(sum([p.numel() for p in state.model.parameters()]))
                    do_training(member_config, state, device_id)
                    model = state.model
                else:
                    print(
                        f"Model for {hash} successfully deserialized, using its state dict."
                    )
                    model = deserialized_model.model
            finally:
                print("Releasing distributed lock explicitly.")
                distributed_train_run_lock.release()

            trained_members.add(stable_hash(member_config))

            model.eval()
            members.append(model)
        if len(trained_members) < len(ensemble_config.members):
            print("Waiting for members that are probably being trained elsewhere:")
            print(all_member_hashes.difference(trained_members))
            print("With train config hashes:")
            print(
                [
                    train_config_hash_dict[hash]
                    for hash in all_member_hashes.difference(trained_members)
                ]
            )
            time.sleep(1)

    return Ensemble(
        member_configs=ensemble_config.members, members=members, n_members=len(members)
    )


def is_ensemble_serialized(ensemble_config: EnsembleConfig):
    for member_config in ensemble_config.members:
        if not is_serialized(member_config):
            return False

    return True


def ensemble_mean_prediction(ensemble: Ensemble, x: torch.Tensor):
    outputs = []
    for member in ensemble.members:
        output = member(x)
        outputs.append(output["predictions"])
    all_outputs = torch.stack(outputs, dim=0)
    predictions = torch.mean(all_outputs, dim=0)
    return dict(
        predictions=predictions,
        logits=torch.log(predictions + torch.finfo(torch.float32).eps),
    )
