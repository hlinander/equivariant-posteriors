from typing import Callable, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEpochSpec
from lib.train import load_or_create_state
from lib.train import do_training, validate
from lib.serialization import DeserializeConfig, get_checkpoint_path, is_serialized
from lib.model_factory import get_factory


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
        checkpoint_path, _ = get_checkpoint_path(member_config.train_config)
        checkpoint_files += [
            (member_config, path.absolute())
            for path in checkpoint_path.parent.glob(f"{checkpoint_path.stem}*")
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
    print("Will try to load ensemble checkpoints from:")
    for member_config in ensemble_config.members:
        print(get_checkpoint_path(member_config.train_config)[0].as_posix())
    print("Loading or training ensemble...")
    for member_config in ensemble_config.members:
        state = load_or_create_state(member_config, device_id)
        print(sum([p.numel() for p in state.model.parameters()]))
        do_training(member_config, state, device_id)
        state.model.eval()
        members.append(state.model)

    return Ensemble(
        member_configs=ensemble_config.members, members=members, n_members=len(members)
    )
