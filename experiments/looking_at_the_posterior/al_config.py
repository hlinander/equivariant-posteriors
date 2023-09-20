from typing import List
from dataclasses import dataclass
import numpy as np

from lib.ensemble import EnsembleConfig


@dataclass
class ALConfig:
    ensemble_config: EnsembleConfig
    uq_calibration_data_config: object
    data_validation_config: object
    data_pool_config: object
    aquisition_method: str
    seed: int = 42
    n_start: int = 100
    n_end: int = 3000
    n_steps: int = 20
    n_epochs_per_step: int = 50
    n_members: int = 5

    def serialize_human(self):
        return {
            "ensemble_config": self.ensemble_config.serialize_human(),
            "uq_calibration_data_config": self.ensemble_config.serialize_human(),
            "data_validation_config": self.ensemble_config.serialize_human(),
            "data_pool_config": self.ensemble_config.serialize_human(),
            "n_start": self.n_start,
            "n_end": self.n_end,
            "n_steps": self.n_steps,
        }


@dataclass
class ALStep:
    al_config: ALConfig
    ensemble: object
    pool_ids: List[int]
    rng: np.random.Generator
    step: int
