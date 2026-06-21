"""Real NxN determinant with matrix-shaped input and RAW log|det| target, for
the elimination-rollout model.

Unlike DataRealDet (flattened input, standardized log|det| target for the
regression baseline), this keeps the matrix shape (n, n) -- the rollout model
operates on rows -- and an unstandardized log|det| target, so the model's
Sum_d log|diag| readout can be compared directly. Gaussian-iid entries, exact
target via slogdet; train set is a deterministic seeded draw, val a disjoint
fixed draw.
"""
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy

_VAL_SEED_OFFSET = 1 << 40


@dataclass(frozen=True)
class DataRealDetMatrixConfig:
    n: int = 3
    n_train: int = 100000
    n_val: int = 20000
    sigma: float = 1.0
    seed: int = 0
    validation: bool = False

    @property
    def n_samples(self):
        return self.n_val if self.validation else self.n_train

    def serialize_human(self):
        return {**self.__dict__, "n_samples": self.n_samples}


class DataRealDetMatrix(torch.utils.data.Dataset):
    def __init__(self, data_config: DataRealDetMatrixConfig):
        n = data_config.n
        sigma = data_config.sigma
        if data_config.validation:
            g = torch.Generator().manual_seed(data_config.seed + _VAL_SEED_OFFSET)
            n_draw = data_config.n_val
        else:
            g = torch.Generator().manual_seed(data_config.seed)
            n_draw = data_config.n_train

        matrices = torch.randn(n_draw, n, n, generator=g) * sigma
        sign, logabs = torch.linalg.slogdet(matrices)

        self.xs = matrices  # (n_samples, n, n)
        self.ys = logabs.unsqueeze(1)  # (n_samples, 1) raw log|det|
        self.signs = (sign > 0).float().unsqueeze(1)
        self.sample_ids = torch.arange(n_draw, dtype=torch.int32)

    @staticmethod
    def data_spec(config):
        return DataSpec(
            input_shape=torch.Size([config.n, config.n]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([1]),
        )

    def __getitem__(self, idx):
        return create_sample_legacy(self.xs[idx], self.ys[idx], self.sample_ids[idx])

    def __getitems__(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.xs.device)
        return dict(
            input=self.xs[idx],
            target=self.ys[idx],
            sample_id=self.sample_ids[idx],
        )

    def to(self, device):
        self.xs = self.xs.to(device)
        self.ys = self.ys.to(device)
        self.signs = self.signs.to(device)
        self.sample_ids = self.sample_ids.to(device)
        return self

    @staticmethod
    def collate_fn(batch):
        return batch

    def __len__(self):
        return self.xs.shape[0]
