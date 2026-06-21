"""Iterative elimination rollout: compute log|det| of a matrix by a sequence of
determinant-preserving elementary row operations, reading off Sum_d log|diag|.

Elementary row operations as the "language": for each column k, optionally pick
a pivot row and swap it up (row swap flips det sign but not |det|, so it's free
for the magnitude target), then for each below-diagonal position (i, k) a
weight-tied step MLP reads the current matrix and predicts a multiplier c; the
update row_i <- row_i - c * row_k has det = 1 for ANY c, so the determinant is
exactly conserved through the rollout. The model learns c that zeros A[i,k]
(= A[i,k]/A[k,k]) so the final matrix is upper-triangular and the fixed readout
Sum_d log|A[d,d]| equals log|det A|.

Design knobs (see config): input featurization (raw/log/both -- the multiplier
is a ratio, log -> subtraction), multiplier output param (linear vs log-space,
the division fix at scale), teacher forcing / scheduled sampling (drift), and
pivoting (none/partial/learned) -- partial pivoting bounds |c|<=1, fixing the
heavy-tailed multiplier targets that grow with N without it; learned pivoting is
the discrete strategy decision, supervised against the partial-pivot argmax.
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
    input_features: str = "raw"  # "raw" | "log" | "both"
    multiplier_param: str = "linear"  # "linear" | "log"
    teacher_mode: str = "off"  # "off" | "on" | "anneal"
    teacher_anneal_steps: int = 10000
    pivot: str = "none"  # "none" | "partial" (oracle argmax) | "learned"
    # Greedy residual refinement: after the base pass, take this many extra
    # steps, each eliminating the current largest strict-lower entry. Every step
    # is det-preserving, so more steps only chip away residual (test-time
    # compute). config value is used at training time; rollout(refine_steps=R)
    # overrides it for the eval R-sweep.
    refine_steps: int = 0

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
        self.feat_dim = feat_dims[config.input_features]

        # multiplier step module: features + one-hot(i) + one-hot(k) -> c
        in_dim = self.feat_dim + 2 * self.n
        layers = [torch.nn.Linear(in_dim, config.hidden)]
        for _ in range(config.depth - 1):
            layers.append(torch.nn.Linear(config.hidden, config.hidden))
        self.layers = torch.nn.ModuleList(layers)
        self.head = torch.nn.Linear(config.hidden, 2 if config.multiplier_param == "log" else 1)

        # pivot scorer (learned pivoting only): features + one-hot(k) -> score per row
        if config.pivot == "learned":
            pin = self.feat_dim + self.n
            players = [torch.nn.Linear(pin, config.hidden)]
            for _ in range(config.depth - 1):
                players.append(torch.nn.Linear(config.hidden, config.hidden))
            self.pivot_layers = torch.nn.ModuleList(players)
            self.pivot_head = torch.nn.Linear(config.hidden, self.n)

        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long))

    def _featurize(self, a):
        b = a.shape[0]
        flat = a.reshape(b, -1)
        mode = self.config.input_features
        if mode == "raw":
            return flat
        logf = torch.log(flat.abs().clamp_min(self.config.eps))
        sgn = torch.sign(flat)
        if mode == "log":
            return torch.cat([logf, sgn], dim=1)
        return torch.cat([flat, logf, sgn], dim=1)

    def _onehot(self, b, idx, device, dtype):
        o = torch.zeros(b, self.n, device=device, dtype=dtype)
        o[:, idx] = 1.0
        return o

    def _multiplier_from_onehot(self, feats, oi, ok):
        x = torch.cat([feats, oi, ok], dim=1)
        for layer in self.layers:
            x = self.activation(layer(x))
        out = self.head(x)
        if self.config.multiplier_param == "log":
            u = out[:, 0].clamp(-15.0, 15.0)
            return torch.exp(u) * torch.tanh(out[:, 1])
        return out.squeeze(-1)

    def _multiplier(self, a, i, k, feats):
        b = a.shape[0]
        return self._multiplier_from_onehot(
            feats, self._onehot(b, i, a.device, a.dtype),
            self._onehot(b, k, a.device, a.dtype),
        )

    def _multiplier_idx(self, a, i_idx, k_idx, feats):
        """Per-sample multiplier for index tensors i_idx, k_idx (shape (B,))."""
        oi = torch.nn.functional.one_hot(i_idx, self.n).to(a.dtype)
        ok = torch.nn.functional.one_hot(k_idx, self.n).to(a.dtype)
        return self._multiplier_from_onehot(feats, oi, ok)

    def _refine_step(self, a, oracle):
        """One greedy step: eliminate the current largest strict-lower entry
        (per sample). Returns (new_a, step_mult_loss_or_None)."""
        b, n, _ = a.shape
        bidx = torch.arange(b, device=a.device)
        # restrict argmax to strict-lower positions (i>k); others -> -1 so a
        # near-triangular matrix can't select the diagonal (which would self-
        # eliminate to 0/0).
        valid = torch.tril(torch.ones(n, n, device=a.device), diagonal=-1).reshape(1, n * n).bool()
        lower = torch.tril(a, diagonal=-1).abs().reshape(b, n * n).masked_fill(~valid, -1.0)
        flat = lower.argmax(dim=1)
        i_idx = flat // n
        k_idx = flat % n
        c_oracle = a[bidx, i_idx, k_idx] / a[bidx, k_idx, k_idx]
        step_loss = None
        if oracle:
            c = c_oracle
        else:
            c = self._multiplier_idx(a, i_idx, k_idx, self._featurize(a))
            step_loss = torch.nn.functional.smooth_l1_loss(c, c_oracle.detach())
        delta = -c.unsqueeze(1) * a[bidx, k_idx, :]  # (B, n) pivot rows per sample
        upd = torch.zeros_like(a)
        upd[bidx, i_idx, :] = delta
        return a + upd, step_loss

    def _pivot_scores(self, a, k, feats):
        b = a.shape[0]
        ok = self._onehot(b, k, a.device, a.dtype)
        x = torch.cat([feats, ok], dim=1)
        for layer in self.pivot_layers:
            x = self.activation(layer(x))
        scores = self.pivot_head(x)  # (b, n)
        # rows above k are not valid pivots
        mask = torch.arange(self.n, device=a.device) < k
        return scores.masked_fill(mask.unsqueeze(0), float("-inf"))

    def _swap_rows(self, a, k, p):
        """Per-sample swap of rows k and p[b] (differentiable gather)."""
        b, n, _ = a.shape
        perm = torch.arange(n, device=a.device).unsqueeze(0).expand(b, -1).clone()
        bidx = torch.arange(b, device=a.device)
        old_k = perm[bidx, k].clone()
        old_p = perm[bidx, p].clone()
        perm[bidx, k] = old_p
        perm[bidx, p] = old_k
        return torch.gather(a, 1, perm.unsqueeze(-1).expand(-1, -1, n))

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

    def rollout(self, a, oracle=False, teacher_ratio=0.0, collect_resid=False,
                refine_steps=None):
        """Returns (logdet, lower_tri_sq, final, resid, mult_loss, pivot_loss)."""
        n = self.n
        if refine_steps is None:
            refine_steps = self.config.refine_steps
        resid = []
        mult_loss = a.new_zeros(())
        pivot_loss = a.new_zeros(())
        n_steps = 0
        n_piv = 0
        for k in range(n - 1):
            if self.config.pivot != "none":
                target_p = a[:, k:, k].abs().argmax(dim=1) + k  # partial-pivot argmax
                if self.config.pivot == "learned" and not oracle:
                    scores = self._pivot_scores(a, k, self._featurize(a))
                    pivot_loss = pivot_loss + torch.nn.functional.cross_entropy(
                        scores, target_p.detach()
                    )
                    n_piv += 1
                    model_p = scores.argmax(dim=1)
                    p = target_p if teacher_ratio >= 1.0 else model_p
                else:
                    p = target_p
                a = self._swap_rows(a, k, p)
            for i in range(k + 1, n):
                feats = self._featurize(a)
                c_oracle = a[:, i, k] / a[:, k, k]
                if oracle:
                    c_used = c_oracle
                else:
                    c_model = self._multiplier(a, i, k, feats)
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
        for _ in range(refine_steps):
            a, step_loss = self._refine_step(a, oracle)
            if step_loss is not None:
                mult_loss = mult_loss + step_loss
                n_steps += 1
            if collect_resid:
                resid.append(torch.tril(a, diagonal=-1).abs().amax(dim=(1, 2)).mean())
        if n_steps > 0:
            mult_loss = mult_loss / n_steps
        if n_piv > 0:
            pivot_loss = pivot_loss / n_piv
        diag = a.diagonal(dim1=1, dim2=2)
        logdet = torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1)
        lower_tri_sq = torch.tril(a, diagonal=-1).pow(2).sum(dim=(1, 2))
        return logdet, lower_tri_sq, a, resid, mult_loss, pivot_loss

    @torch.no_grad()
    def trace_rollout(self, a, refine_steps=None):
        """Single-example (batch=1) step-by-step trace for inspection. Free
        rollout (model's own pivots/multipliers). Returns a dict with the
        initial matrix, an ordered list of step records, the final matrix, and
        the predicted log|det| readout."""
        assert a.shape[0] == 1
        n = self.n
        if refine_steps is None:
            refine_steps = self.config.refine_steps
        steps = []
        initial = a.clone()
        for k in range(n - 1):
            if self.config.pivot != "none":
                target_p = int(a[:, k:, k].abs().argmax(dim=1).item()) + k
                if self.config.pivot == "learned":
                    p = int(self._pivot_scores(a, k, self._featurize(a)).argmax(dim=1).item())
                else:
                    p = target_p
                if p != k:
                    a = self._swap_rows(a, k, a.new_tensor([p], dtype=torch.long))
                steps.append(dict(type="pivot", k=k, p=p, target_p=target_p,
                                  matrix=a[0].clone()))
            for i in range(k + 1, n):
                feats = self._featurize(a)
                c_oracle = float((a[:, i, k] / a[:, k, k]).item())
                c_model = float(self._multiplier(a, i, k, feats).item())
                upd = torch.zeros_like(a)
                upd[:, i, :] = -c_model * a[:, k, :]
                a = a + upd
                steps.append(dict(type="elim", k=k, i=i, c_model=c_model,
                                  c_oracle=c_oracle, resid=float(a[:, i, k].abs().item()),
                                  matrix=a[0].clone()))
        valid = torch.tril(torch.ones(n, n, device=a.device), diagonal=-1).reshape(1, n * n).bool()
        for _ in range(refine_steps):
            lower = torch.tril(a, diagonal=-1).abs().reshape(1, n * n).masked_fill(~valid, -1.0)
            flat = int(lower.argmax(dim=1).item())
            i, k = flat // n, flat % n
            c_oracle = float((a[:, i, k] / a[:, k, k]).item())
            c_model = float(self._multiplier_idx(
                a, a.new_tensor([i], dtype=torch.long),
                a.new_tensor([k], dtype=torch.long), self._featurize(a)).item())
            before = float(a[:, i, k].abs().item())
            a, _ = self._refine_step(a, oracle=False)
            steps.append(dict(type="refine", k=k, i=i, c_model=c_model,
                              c_oracle=c_oracle, resid_before=before,
                              resid=float(a[:, i, k].abs().item()),
                              lower_rms=float(torch.tril(a, diagonal=-1).pow(2).sum().sqrt().item()),
                              matrix=a[0].clone()))
        diag = a.diagonal(dim1=1, dim2=2)
        pred = float(torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1).item())
        return dict(initial=initial[0], steps=steps, final=a[0], pred_logdet=pred)

    def forward(self, batch):
        a = batch["input"]
        tr = self._teacher_ratio()
        logdet, lower_tri_sq, _, _, mult_loss, pivot_loss = self.rollout(
            a, teacher_ratio=tr
        )
        if self.training:
            self.train_steps += 1
        return dict(
            logits=logdet.unsqueeze(-1),
            lower_tri_sq=lower_tri_sq,
            mult_loss=mult_loss,
            pivot_loss=pivot_loss,
            predictions=logdet.detach().unsqueeze(-1),
        )
