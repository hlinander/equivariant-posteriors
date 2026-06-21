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
    # Output parametrization of the multiplier. "log": head -> (log|c|, sign),
    # c = sign*exp(log|c|), so the ratio A[i,k]/A[k,k] is a subtraction at the
    # head (linear) regardless of how summed the current entries are -- the
    # division fix that log *input* alone can't give past the first column.
    multiplier_param: str = "linear"  # "linear" | "log"
    # Teacher forcing of the applied multiplier during training: "off" (free),
    # "on" (always apply the oracle c), "anneal" (1->0 over teacher_anneal_steps,
    # scheduled sampling). Per-step supervision against the oracle is applied
    # regardless; teacher forcing only changes which c drives the next state.
    teacher_mode: str = "off"  # "off" | "on" | "anneal"
    teacher_anneal_steps: int = 10000

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
        out_dim = 2 if config.multiplier_param == "log" else 1
        self.head = torch.nn.Linear(config.hidden, out_dim)
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long))

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
        out = self.head(x)
        if self.config.multiplier_param == "log":
            u = out[:, 0].clamp(-15.0, 15.0)  # log|c|
            s = out[:, 1]  # sign logit
            return torch.exp(u) * torch.tanh(s)
        return out.squeeze(-1)

    def _teacher_ratio(self):
        if not self.training:
            return 0.0
        mode = self.config.teacher_mode
        if mode == "on":
            return 1.0
        if mode == "anneal":
            t = float(self.train_steps.item())
            return max(0.0, 1.0 - t / max(1, self.config.teacher_anneal_steps))
        return 0.0

    def rollout(self, a, oracle=False, teacher_ratio=0.0, collect_resid=False):
        """Returns (logdet, lower_tri_sq, final_matrix, per_step_residual,
        multiplier_loss). multiplier_loss is the per-step smooth-L1 between the
        model multiplier and the oracle c=A[i,k]/A[k,k] on the current state."""
        n = self.n
        resid = []
        mult_loss = a.new_zeros(())
        n_steps = 0
        for k in range(n - 1):
            for i in range(k + 1, n):
                c_oracle = a[:, i, k] / a[:, k, k]
                if oracle:
                    c_used = c_oracle
                else:
                    c_model = self._multiplier(a, i, k)
                    mult_loss = mult_loss + torch.nn.functional.smooth_l1_loss(
                        c_model, c_oracle.detach()
                    )
                    n_steps += 1
                    if teacher_ratio >= 1.0:
                        c_used = c_oracle.detach()
                    elif teacher_ratio > 0.0:
                        mask = (torch.rand_like(c_model) < teacher_ratio).float()
                        c_used = mask * c_oracle.detach() + (1.0 - mask) * c_model
                    else:
                        c_used = c_model
                upd = torch.zeros_like(a)
                upd[:, i, :] = -c_used.unsqueeze(1) * a[:, k, :]
                a = a + upd
                if collect_resid:
                    resid.append(a[:, i, k].abs().mean())
        if n_steps > 0:
            mult_loss = mult_loss / n_steps
        diag = a.diagonal(dim1=1, dim2=2)
        logdet = torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1)
        lower_tri_sq = torch.tril(a, diagonal=-1).pow(2).sum(dim=(1, 2))
        return logdet, lower_tri_sq, a, resid, mult_loss

    def forward(self, batch):
        a = batch["input"]
        tr = self._teacher_ratio()
        logdet, lower_tri_sq, _, _, mult_loss = self.rollout(a, teacher_ratio=tr)
        if self.training:
            self.train_steps += 1
        return dict(
            logits=logdet.unsqueeze(-1),
            lower_tri_sq=lower_tri_sq,
            mult_loss=mult_loss,
            predictions=logdet.detach().unsqueeze(-1),
        )
