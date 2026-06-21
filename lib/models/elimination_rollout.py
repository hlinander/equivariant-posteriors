"""Iterative elimination rollout: compute log|det| of a matrix by a sequence of
determinant-preserving elementary row operations, reading off Sum_d log|diag|.

The step that predicts each operation has two architectures (config.step_arch):

- "mlp": features of the whole flattened matrix + one-hot(i) + one-hot(k) -> c.
  The one-hots must gate which of the n^2 entries to read (a binding problem an
  MLP does poorly), which caps multiplier precision as N grows.

- "transformer": rows are tokens; self-attention does the entry/row selection
  natively, and the multiplier for (i,k) is read from the contextual row
  embeddings h_i, h_k. Permutation-structured over rows and (in principle)
  size-agnostic -- the natural fit for "predict the next elementary operator".

Other knobs: input featurization (mlp only), multiplier output param
(linear vs log-space ratio), teacher forcing / scheduled sampling, pivoting
(none/partial/learned), and greedy residual refinement (test-time compute).
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
    eps: float = 1e-6
    input_features: str = "raw"  # mlp only: "raw" | "log" | "both"
    multiplier_param: str = "linear"  # "linear" | "log"
    teacher_mode: str = "off"  # "off" | "on" | "anneal"
    teacher_anneal_steps: int = 10000
    pivot: str = "none"  # "none" | "partial" | "learned"
    refine_steps: int = 0
    step_arch: str = "mlp"  # "mlp" | "transformer"
    num_heads: int = 4  # transformer only

    def serialize_human(self):
        return self.__dict__


class EliminationRollout(torch.nn.Module):
    def __init__(self, config: EliminationRolloutConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.n = data_spec.input_shape[0]
        self.activation = get_activation(config.activation)
        nn2 = self.n * self.n
        out_dim = 2 if config.multiplier_param == "log" else 1

        if config.step_arch == "mlp":
            feat_dims = {"raw": nn2, "log": 2 * nn2, "both": 3 * nn2}
            if config.input_features not in feat_dims:
                raise ValueError(f"unknown input_features {config.input_features}")
            self.feat_dim = feat_dims[config.input_features]
            in_dim = self.feat_dim + 2 * self.n
            layers = [torch.nn.Linear(in_dim, config.hidden)]
            for _ in range(config.depth - 1):
                layers.append(torch.nn.Linear(config.hidden, config.hidden))
            self.layers = torch.nn.ModuleList(layers)
            self.head = torch.nn.Linear(config.hidden, out_dim)
            if config.pivot == "learned":
                pin = self.feat_dim + self.n
                players = [torch.nn.Linear(pin, config.hidden)]
                for _ in range(config.depth - 1):
                    players.append(torch.nn.Linear(config.hidden, config.hidden))
                self.pivot_layers = torch.nn.ModuleList(players)
                self.pivot_head = torch.nn.Linear(config.hidden, self.n)
        elif config.step_arch == "transformer":
            if config.pivot == "learned":
                raise NotImplementedError("learned pivot not wired for transformer step")
            # row token: [entries, log|entries|, sign] -> hidden, + row positional
            self.row_embed = torch.nn.Linear(3 * self.n, config.hidden)
            self.row_pos = torch.nn.Embedding(self.n, config.hidden)
            enc_layer = torch.nn.TransformerEncoderLayer(
                d_model=config.hidden, nhead=config.num_heads,
                dim_feedforward=config.hidden * 2, dropout=0.0,
                activation=config.activation, batch_first=True, norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(
                enc_layer, num_layers=config.depth,
                norm=torch.nn.LayerNorm(config.hidden),  # final norm (pre-LN stack)
                enable_nested_tensor=False,
            )
            self.mult_head = torch.nn.Sequential(
                torch.nn.Linear(2 * config.hidden, config.hidden),
                torch.nn.GELU(),
                torch.nn.Linear(config.hidden, out_dim),
            )
        else:
            raise ValueError(f"unknown step_arch {config.step_arch}")

        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long))

    # ---- feature helpers ----
    def _featurize(self, a):  # mlp path: flattened whole-matrix features
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

    def _encode_rows(self, a):  # transformer path: contextual row embeddings
        logf = torch.log(a.abs().clamp_min(self.config.eps))
        rf = torch.cat([a, logf, torch.sign(a)], dim=-1)  # (B, n, 3n)
        pos = self.row_pos(torch.arange(self.n, device=a.device)).unsqueeze(0)
        return self.encoder(self.row_embed(rf) + pos)  # (B, n, hidden)

    def _onehot(self, b, idx, device, dtype):
        o = torch.zeros(b, self.n, device=device, dtype=dtype)
        o[:, idx] = 1.0
        return o

    def _mult_from_raw(self, raw):
        if self.config.multiplier_param == "log":
            u = raw[:, 0].clamp(-15.0, 15.0)
            return torch.exp(u) * torch.tanh(raw[:, 1])
        return raw.squeeze(-1)

    # ---- multiplier (scalar i,k for the base pass) ----
    def _multiplier(self, a, i, k):
        if self.config.step_arch == "mlp":
            b = a.shape[0]
            x = torch.cat([self._featurize(a), self._onehot(b, i, a.device, a.dtype),
                           self._onehot(b, k, a.device, a.dtype)], dim=1)
            for layer in self.layers:
                x = self.activation(layer(x))
            return self._mult_from_raw(self.head(x))
        h = self._encode_rows(a)
        return self._mult_from_raw(self.mult_head(torch.cat([h[:, i], h[:, k]], dim=1)))

    # ---- multiplier (per-sample index tensors, for refinement) ----
    def _multiplier_idx(self, a, i_idx, k_idx):
        b = a.shape[0]
        bidx = torch.arange(b, device=a.device)
        if self.config.step_arch == "mlp":
            oi = torch.nn.functional.one_hot(i_idx, self.n).to(a.dtype)
            ok = torch.nn.functional.one_hot(k_idx, self.n).to(a.dtype)
            x = torch.cat([self._featurize(a), oi, ok], dim=1)
            for layer in self.layers:
                x = self.activation(layer(x))
            return self._mult_from_raw(self.head(x))
        h = self._encode_rows(a)
        return self._mult_from_raw(
            self.mult_head(torch.cat([h[bidx, i_idx], h[bidx, k_idx]], dim=1))
        )

    def _pivot_scores(self, a, k):
        b = a.shape[0]
        ok = self._onehot(b, k, a.device, a.dtype)
        x = torch.cat([self._featurize(a), ok], dim=1)
        for layer in self.pivot_layers:
            x = self.activation(layer(x))
        scores = self.pivot_head(x)
        mask = torch.arange(self.n, device=a.device) < k
        return scores.masked_fill(mask.unsqueeze(0), float("-inf"))

    def _swap_rows(self, a, k, p):
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

    def _refine_step(self, a, oracle):
        b, n, _ = a.shape
        bidx = torch.arange(b, device=a.device)
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
            c = self._multiplier_idx(a, i_idx, k_idx)
            step_loss = torch.nn.functional.smooth_l1_loss(c, c_oracle.detach())
        delta = -c.unsqueeze(1) * a[bidx, k_idx, :]
        upd = torch.zeros_like(a)
        upd[bidx, i_idx, :] = delta
        return a + upd, step_loss

    def rollout(self, a, oracle=False, teacher_ratio=0.0, collect_resid=False,
                refine_steps=None):
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
                target_p = a[:, k:, k].abs().argmax(dim=1) + k
                if self.config.pivot == "learned" and not oracle:
                    scores = self._pivot_scores(a, k)
                    pivot_loss = pivot_loss + torch.nn.functional.cross_entropy(
                        scores, target_p.detach()
                    )
                    n_piv += 1
                    p = target_p if teacher_ratio >= 1.0 else scores.argmax(dim=1)
                else:
                    p = target_p
                a = self._swap_rows(a, k, p)
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
                    p = int(self._pivot_scores(a, k).argmax(dim=1).item())
                else:
                    p = target_p
                if p != k:
                    a = self._swap_rows(a, k, a.new_tensor([p], dtype=torch.long))
                steps.append(dict(type="pivot", k=k, p=p, target_p=target_p,
                                  matrix=a[0].clone()))
            for i in range(k + 1, n):
                c_oracle = float((a[:, i, k] / a[:, k, k]).item())
                c_model = float(self._multiplier(a, i, k).item())
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
                a, a.new_tensor([i], dtype=torch.long), a.new_tensor([k], dtype=torch.long)).item())
            before = float(a[:, i, k].abs().item())
            a, _ = self._refine_step(a, oracle=False)
            steps.append(dict(type="refine", k=k, i=i, c_model=c_model, c_oracle=c_oracle,
                              resid_before=before, resid=float(a[:, i, k].abs().item()),
                              lower_rms=float(torch.tril(a, diagonal=-1).pow(2).sum().sqrt().item()),
                              matrix=a[0].clone()))
        diag = a.diagonal(dim1=1, dim2=2)
        pred = float(torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1).item())
        return dict(initial=initial[0], steps=steps, final=a[0], pred_logdet=pred)

    def forward(self, batch):
        a = batch["input"]
        tr = self._teacher_ratio()
        logdet, lower_tri_sq, _, _, mult_loss, pivot_loss = self.rollout(a, teacher_ratio=tr)
        if self.training:
            self.train_steps += 1
        return dict(
            logits=logdet.unsqueeze(-1),
            lower_tri_sq=lower_tri_sq,
            mult_loss=mult_loss,
            pivot_loss=pivot_loss,
            predictions=logdet.detach().unsqueeze(-1),
        )
