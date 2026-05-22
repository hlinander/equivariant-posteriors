import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy


@dataclass(frozen=True)
class DataSparseParityConfig:
    d: int = 40
    k: int = 3
    n_samples: int = 1024
    seed: int = 0
    subset_seed: int = 0

    def serialize_human(self):
        return self.__dict__


class DataSparseParity(torch.utils.data.Dataset):
    def __init__(self, data_config: DataSparseParityConfig):
        d = data_config.d
        k = data_config.k

        # Select parity subset deterministically from (d, k, subset_seed)
        subset_rng = torch.Generator()
        subset_rng.manual_seed(data_config.subset_seed)
        self.S = torch.randperm(d, generator=subset_rng)[:k]

        # Generate data samples
        rng = torch.Generator()
        rng.manual_seed(data_config.seed)
        self.xs = 2 * torch.randint(
            0, 2, (data_config.n_samples, d), generator=rng
        ).float() - 1

        # y = product of x_i for i in S, mapped to {0, 1}
        parity = self.xs[:, self.S].prod(dim=1)  # in {-1, +1}
        self.ys = ((parity + 1) / 2).long()  # map -1->0, +1->1
        self.sample_ids = torch.arange(data_config.n_samples, dtype=torch.int32)

    @staticmethod
    def data_spec(config):
        return DataSpec(
            input_shape=torch.Size([config.d]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([2]),
        )

    def __getitem__(self, idx):
        return create_sample_legacy(self.xs[idx], self.ys[idx], self.sample_ids[idx])

    def __len__(self):
        return self.xs.shape[0]
