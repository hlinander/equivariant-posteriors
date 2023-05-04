from typing import Callable
from dataclasses import dataclass
import torch
from lib.train_dataclasses import TrainRun
from lib.train import load_or_create_state
from lib.train import do_training


@dataclass
class EnsembleConfig:
    members: list[TrainRun]


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


def create_ensemble(ensemble_config: EnsembleConfig, device_id):
    members = []
    for member_config in ensemble_config.members:
        state = load_or_create_state(member_config, device_id)
        do_training(member_config, state, device_id)
        members.append(state.model)

    return Ensemble(
        member_configs=ensemble_config.members, members=members, n_members=len(members)
    )