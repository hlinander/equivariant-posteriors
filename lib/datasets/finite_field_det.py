import torch
import itertools
import math
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy


@dataclass(frozen=True)
class DataFiniteFieldDetConfig:
    n: int = 3
    p: int = 5
    n_samples: int = 10000
    seed: int = 0

    def serialize_human(self):
        return self.__dict__


def _det_mod_p_batched(matrices, p):
    """Compute determinant mod p for a batch of integer matrices using Leibniz formula.

    Args:
        matrices: (batch, n, n) integer tensor with entries in {0, ..., p-1}
        p: prime modulus

    Returns:
        (batch,) long tensor of determinants mod p
    """
    n = matrices.shape[1]
    det = torch.zeros(matrices.shape[0], dtype=torch.long)
    for perm in itertools.permutations(range(n)):
        # Compute sign of permutation
        sign = 1
        perm_list = list(perm)
        for i in range(n):
            while perm_list[i] != i:
                j = perm_list[i]
                perm_list[i], perm_list[j] = perm_list[j], perm_list[i]
                sign *= -1
        # Vectorized product over batch: product of matrices[:, i, perm[i]] for all i
        product = torch.ones(matrices.shape[0], dtype=torch.long)
        for i, j in enumerate(perm):
            product = (product * matrices[:, i, j]) % p
        det = (det + sign * product) % p
    return det % p


class DataFiniteFieldDet(torch.utils.data.Dataset):
    def __init__(self, data_config: DataFiniteFieldDetConfig):
        n = data_config.n
        p = data_config.p
        n_entries = n * n

        rng = torch.Generator()
        rng.manual_seed(data_config.seed)
        # Sample random matrices with entries in {0, ..., p-1}
        entries = torch.randint(0, p, (data_config.n_samples, n_entries), generator=rng)

        # One-hot encode each entry
        one_hot = torch.nn.functional.one_hot(entries.long(), num_classes=p).float()
        self.xs = one_hot.reshape(data_config.n_samples, n_entries * p)

        # Compute determinant mod p for each matrix (vectorized)
        matrices = entries.long().reshape(data_config.n_samples, n, n)
        self.ys = _det_mod_p_batched(matrices, p)

        self.sample_ids = torch.arange(data_config.n_samples, dtype=torch.int32)

    @staticmethod
    def data_spec(config):
        dim = config.n * config.n * config.p
        return DataSpec(
            input_shape=torch.Size([dim]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([config.p]),
        )

    def __getitem__(self, idx):
        return create_sample_legacy(self.xs[idx], self.ys[idx], self.sample_ids[idx])

    def __len__(self):
        return self.xs.shape[0]
