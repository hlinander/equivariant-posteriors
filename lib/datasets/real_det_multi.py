"""Multitask real-matrix dataset for the operator-algebra transformer.

Same Gaussian-iid matrices as DataRealDetMatrix, but each sample carries a task
id so one model can be trained on several "sentences" sharing the operator
vocabulary:
    task 0 = DET : reduce A to upper-triangular, read Sum log|diag|
    task 1 = INV : reduce A to identity (Gauss-Jordan); composed ops = A^{-1}

Tasks are assigned deterministically by sample parity (~50/50). The batch dict
gains a `task` field (the model picks the task token and teacher sequence from
it). target carries log|det| (used by the DET metric; INV is scored on
reconstruction in the hook).
"""
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec

_VAL_SEED_OFFSET = 1 << 40


@dataclass(frozen=True)
class DataRealDetMultiConfig:
    n: int = 4
    n_train: int = 100000
    n_val: int = 20000
    sigma: float = 1.0
    seed: int = 0
    validation: bool = False
    tasks: tuple = (0, 1)  # which task ids to include

    @property
    def n_samples(self):
        return self.n_val if self.validation else self.n_train

    def serialize_human(self):
        return {**self.__dict__, "n_samples": self.n_samples}


class DataRealDetMulti(torch.utils.data.Dataset):
    def __init__(self, data_config: DataRealDetMultiConfig):
        n = data_config.n
        if data_config.validation:
            g = torch.Generator().manual_seed(data_config.seed + _VAL_SEED_OFFSET)
            n_draw = data_config.n_val
        else:
            g = torch.Generator().manual_seed(data_config.seed)
            n_draw = data_config.n_train

        self.xs = torch.randn(n_draw, n, n, generator=g) * data_config.sigma
        _, logabs = torch.linalg.slogdet(self.xs)
        self.ys = logabs.unsqueeze(1)
        # round-robin over the included tasks (deterministic, balanced)
        tasks = torch.tensor(data_config.tasks, dtype=torch.long)
        self.task = tasks[torch.arange(n_draw) % len(tasks)]
        self.sample_ids = torch.arange(n_draw, dtype=torch.int32)

    @staticmethod
    def data_spec(config):
        return DataSpec(
            input_shape=torch.Size([config.n, config.n]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([1]),
        )

    def __getitem__(self, idx):
        return dict(
            input=self.xs[idx], target=self.ys[idx], task=self.task[idx],
            sample_id=self.sample_ids[idx],
        )

    def __getitems__(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.xs.device)
        return dict(
            input=self.xs[idx], target=self.ys[idx], task=self.task[idx],
            sample_id=self.sample_ids[idx],
        )

    def to(self, device):
        self.xs = self.xs.to(device)
        self.ys = self.ys.to(device)
        self.task = self.task.to(device)
        self.sample_ids = self.sample_ids.to(device)
        return self

    @staticmethod
    def collate_fn(batch):
        return batch

    def __len__(self):
        return self.xs.shape[0]
