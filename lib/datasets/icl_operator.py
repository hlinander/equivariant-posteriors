"""In-context operator learning. Per sequence a latent operator W (d×d) is drawn
from a function class (spectrum); the context is example pairs (x_i, W x_i) and
the query asks for W x_q. The model must infer W from the examples (ICL) and
apply it. Used by the operator-algebra transformer's [ICL] sentence, where W is
emitted as a sequence of rank-1 SVD-component operators (test-time compute).

Function classes (spectrum of W = U diag(s) V^T, U,V random orthogonal):
    full       : s ~ uniform-ish (flat) — needs all d rank-1 ops
    orthogonal : s = 1 — rotation/reflection
    lowrank    : s has only `rank` nonzeros — few ops suffice
    powerlaw   : s_j = (j+1)^(-alpha) — fast-decaying, compute-adaptive
"""
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec

_VAL_SEED_OFFSET = 1 << 40


@dataclass(frozen=True)
class DataICLOperatorConfig:
    n: int = 4  # vector / operator dim d
    k: int = 8  # number of in-context example pairs
    spectrum: str = "powerlaw"  # full | orthogonal | lowrank | powerlaw
    rank: int = 2  # for lowrank
    alpha: float = 1.0  # for powerlaw
    noise: float = 0.0  # observation noise on y
    n_train: int = 100000
    n_val: int = 20000
    seed: int = 0
    validation: bool = False

    @property
    def n_samples(self):
        return self.n_val if self.validation else self.n_train

    def serialize_human(self):
        return {**self.__dict__, "n_samples": self.n_samples}


def _spectrum(cfg, B, g, device):
    d = cfg.n
    if cfg.spectrum == "orthogonal":
        s = torch.ones(B, d, device=device)
    elif cfg.spectrum == "lowrank":
        s = torch.zeros(B, d, device=device)
        s[:, : cfg.rank] = 1.0
    elif cfg.spectrum == "powerlaw":
        j = torch.arange(1, d + 1, device=device, dtype=torch.float32)
        s = (j ** (-cfg.alpha)).unsqueeze(0).expand(B, -1).clone()
    else:  # full
        s = 0.5 + torch.rand(B, d, generator=g, device=device)
    return s


def _rand_orthogonal(B, d, g, device):
    a = torch.randn(B, d, d, generator=g, device=device)
    q, r = torch.linalg.qr(a)
    # fix sign so it's deterministic given the gaussian
    sign = torch.sign(torch.diagonal(r, dim1=1, dim2=2))
    return q * sign.unsqueeze(1)


class DataICLOperator(torch.utils.data.Dataset):
    def __init__(self, data_config: DataICLOperatorConfig):
        cfg = data_config
        d, k = cfg.n, cfg.k
        seed = cfg.seed + (_VAL_SEED_OFFSET if cfg.validation else 0)
        n = cfg.n_val if cfg.validation else cfg.n_train
        g = torch.Generator().manual_seed(seed)

        U = _rand_orthogonal(n, d, g, "cpu")
        V = _rand_orthogonal(n, d, g, "cpu")
        s = _spectrum(cfg, n, g, "cpu")
        self.W = U @ torch.diag_embed(s) @ V.transpose(1, 2)  # (n, d, d)

        self.cx = torch.randn(n, k, d, generator=g)  # context inputs (n,k,d)
        cy = torch.einsum("bij,bkj->bki", self.W, self.cx)  # W x_i
        if cfg.noise > 0:
            cy = cy + cfg.noise * torch.randn(n, k, d, generator=g)
        self.cy = cy
        self.qx = torch.randn(n, d, generator=g)  # query
        self.qy = torch.einsum("bij,bj->bi", self.W, self.qx)
        self.sample_ids = torch.arange(n, dtype=torch.int32)

    @staticmethod
    def data_spec(config):
        return DataSpec(
            input_shape=torch.Size([config.n, config.n]),
            target_shape=torch.Size([config.n]),
            output_shape=torch.Size([config.n]),
        )

    def _item(self, idx):
        return dict(
            context_x=self.cx[idx], context_y=self.cy[idx], query_x=self.qx[idx],
            target=self.qy[idx], W=self.W[idx], sample_id=self.sample_ids[idx],
        )

    def __getitem__(self, idx):
        return self._item(idx)

    def __getitems__(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.cx.device)
        return dict(
            context_x=self.cx[idx], context_y=self.cy[idx], query_x=self.qx[idx],
            target=self.qy[idx], W=self.W[idx], sample_id=self.sample_ids[idx],
        )

    def to(self, device):
        for a in ("W", "cx", "cy", "qx", "qy", "sample_ids"):
            setattr(self, a, getattr(self, a).to(device))
        return self

    @staticmethod
    def collate_fn(batch):
        return batch

    def __len__(self):
        return self.cx.shape[0]
