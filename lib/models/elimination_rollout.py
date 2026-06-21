"""Iterative elimination rollout: compute log|det| of a matrix by a sequence of
determinant-preserving elementary row operations, reading off Sum_d log|diag|.

Elementary row operations as the "language": at each below-diagonal position
(i, k) in column-major order, a weight-tied step MLP reads the current matrix
(and one-hot row indices i, k) and predicts a multiplier c; the update
    row_i <- row_i - c * row_k
has det = 1 for ANY c, so the determinant is exactly conserved through the
rollout. The model only has to learn c that zeros A[i,k] (= A[i,k]/A[k,k]) so
the final matrix is upper-triangular and the fixed readout Sum_d log|A[d,d]|
equals log|det A|. Weight-tying across positions/steps is what lets one learned
step generalize across matrix size and rollout length.
"""
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.models.mlp import get_activation


@dataclass(frozen=True)
class EliminationRolloutConfig:
    hidden: int = 256
    depth: int = 2
    activation: str = "gelu"
    eps: float = 1e-6  # floor for log|diag| readout
    # Input featurization for the step module. The multiplier is a ratio
    # A[i,k]/A[k,k]; in log space that's a subtraction (linear), and MLPs are
    # bad at division -- so "log" feeds log|M| + sign(M), "both" adds raw M too
    # (raw is still needed: the row update is arithmetic on raw entries).
    input_features: str = "raw"  # "raw" | "log" | "both"

    def serialize_human(self):
        return self.__dict__


class EliminationRollout(torch.nn.Module):
    def __init__(self, config: EliminationRolloutConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.n = data_spec.input_shape[0]
        self.activation = get_activation(config.activation)
        nn2 = self.n * self.n
        feat_dims = {"raw": nn2, "log": 2 * nn2, "both": 3 * nn2}
        if config.input_features not in feat_dims:
            raise ValueError(f"unknown input_features {config.input_features}")
        in_dim = feat_dims[config.input_features] + 2 * self.n
        layers = [torch.nn.Linear(in_dim, config.hidden)]
        for _ in range(config.depth - 1):
            layers.append(torch.nn.Linear(config.hidden, config.hidden))
        self.layers = torch.nn.ModuleList(layers)
        self.head = torch.nn.Linear(config.hidden, 1)

    def _multiplier(self, a, i, k):
        b = a.shape[0]
        oi = torch.zeros(b, self.n, device=a.device, dtype=a.dtype)
        ok = torch.zeros(b, self.n, device=a.device, dtype=a.dtype)
        oi[:, i] = 1.0
        ok[:, k] = 1.0
        flat = a.reshape(b, -1)
        mode = self.config.input_features
        if mode == "raw":
            feats = [flat]
        elif mode == "log":
            feats = [torch.log(flat.abs().clamp_min(self.config.eps)), torch.sign(flat)]
        else:  # both
            feats = [flat, torch.log(flat.abs().clamp_min(self.config.eps)), torch.sign(flat)]
        x = torch.cat(feats + [oi, ok], dim=1)
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.head(x).squeeze(-1)

    def rollout(self, a, oracle=False, collect_resid=False):
        """Returns (logdet, lower_tri_sq, final_matrix, per_step_residual)."""
        n = self.n
        resid = []
        for k in range(n - 1):
            for i in range(k + 1, n):
                if oracle:
                    c = a[:, i, k] / a[:, k, k]
                else:
                    c = self._multiplier(a, i, k)
                upd = torch.zeros_like(a)
                upd[:, i, :] = -c.unsqueeze(1) * a[:, k, :]
                a = a + upd
                if collect_resid:
                    resid.append(a[:, i, k].abs().mean())
        diag = a.diagonal(dim1=1, dim2=2)
        logdet = torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1)
        lower_tri_sq = torch.tril(a, diagonal=-1).pow(2).sum(dim=(1, 2))
        return logdet, lower_tri_sq, a, resid

    def forward(self, batch):
        a = batch["input"]
        logdet, lower_tri_sq, _, _ = self.rollout(a)
        return dict(
            logits=logdet.unsqueeze(-1),
            lower_tri_sq=lower_tri_sq,
            predictions=logdet.detach().unsqueeze(-1),
        )
