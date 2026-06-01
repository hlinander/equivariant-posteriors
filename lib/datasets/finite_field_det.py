import torch
import itertools
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy


@dataclass(frozen=True)
class DataFiniteFieldDetConfig:
    n: int = 3
    p: int = 5
    frac: float = 0.5  # fraction of total dataset (p^(n*n)) used for training
    seed: int = 0  # controls the train/val split
    seq: bool = False  # True: input shape (n*n, p) for transformers
    validation: bool = False  # True: return held-out complement

    @property
    def total_size(self):
        return self.p ** (self.n * self.n)

    @property
    def n_samples(self):
        n_train = int(self.frac * self.total_size)
        if self.validation:
            return self.total_size - n_train
        return n_train

    def serialize_human(self):
        return {**self.__dict__, "n_samples": self.n_samples, "total_size": self.total_size}


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


def _enumerate_all_matrices(n, p):
    """Enumerate all p^(n*n) matrices over F_p as a (total, n*n) integer tensor."""
    n_entries = n * n
    total = p ** n_entries
    # Build all combinations using base-p digit expansion
    entries = torch.zeros(total, n_entries, dtype=torch.long)
    for col in range(n_entries):
        period = p ** (n_entries - 1 - col)
        entries[:, col] = (torch.arange(total) // period) % p
    return entries


class DataFiniteFieldDet(torch.utils.data.Dataset):
    def __init__(self, data_config: DataFiniteFieldDetConfig):
        n = data_config.n
        p = data_config.p
        n_entries = n * n

        # Enumerate all matrices and shuffle deterministically
        all_entries = _enumerate_all_matrices(n, p)
        rng = torch.Generator()
        rng.manual_seed(data_config.seed)
        perm = torch.randperm(all_entries.shape[0], generator=rng)
        all_entries = all_entries[perm]

        # Split into train / validation
        n_train = int(data_config.frac * all_entries.shape[0])
        if data_config.validation:
            entries = all_entries[n_train:]
        else:
            entries = all_entries[:n_train]

        # One-hot encode each entry
        one_hot = torch.nn.functional.one_hot(entries, num_classes=p).float()
        if data_config.seq:
            self.xs = one_hot  # (n_samples, n*n, p)
        else:
            self.xs = one_hot.reshape(entries.shape[0], n_entries * p)

        # Compute determinant mod p for each matrix (vectorized)
        matrices = entries.reshape(entries.shape[0], n, n)
        self.ys = _det_mod_p_batched(matrices, p)

        self.sample_ids = torch.arange(entries.shape[0], dtype=torch.int32)

    @staticmethod
    def data_spec(config):
        if config.seq:
            return DataSpec(
                input_shape=torch.Size([config.n * config.n, config.p]),
                target_shape=torch.Size([1]),
                output_shape=torch.Size([config.p]),
            )
        return DataSpec(
            input_shape=torch.Size([config.n * config.n * config.p]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([config.p]),
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
        self.sample_ids = self.sample_ids.to(device)
        return self

    @staticmethod
    def collate_fn(batch):
        return batch

    def __len__(self):
        return self.xs.shape[0]
