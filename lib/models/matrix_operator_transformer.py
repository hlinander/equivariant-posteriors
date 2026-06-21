"""Matrix-operator-token transformer: a sequence model over an operator algebra.

Tokens are full matrices. The sequence is [TASK] [A_0] [A_1] ... [A_m], where
each A_t is the matrix state after t predicted operators. At each state position
a causal transformer predicts the NEXT operator as a delta from identity
(M = I + Delta, applied A <- M A) plus a discrete STOP decision; identity
(Delta = 0) is the natural no-op, so over-generating is harmless.

This is the general substrate (single-task DET first): determinant = transform A
to upper-triangular by a sequence of elementary operators, read Sum log|diag|.
The operator family here is the elimination delta (sparse), supervised
teacher-forced against ground-truth Gaussian-elimination operators; the head
predicts a full n x n delta, so it can equally emit dense operators. Trained as
a parallel teacher-forced causal sequence model (no per-step re-encoding / deep
BPTT, unlike the unrolled elimination rollout) -- the scaling fix.
"""
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass(frozen=True)
class MatrixOperatorTransformerConfig:
    hidden: int = 256
    depth: int = 4
    num_heads: int = 8
    eps: float = 1e-6
    max_ops: int = 0  # 0 -> n*(n-1)//2 (fixed elimination length)

    def serialize_human(self):
        return self.__dict__


class MatrixOperatorTransformer(torch.nn.Module):
    def __init__(self, config: MatrixOperatorTransformerConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.n = data_spec.input_shape[0]
        H = config.hidden
        self.nn2 = self.n * self.n
        self.matrix_embed = torch.nn.Linear(self.nn2, H)
        self.task_embed = torch.nn.Parameter(torch.zeros(1, 1, H))
        self.max_ops = config.max_ops or (self.n * (self.n - 1) // 2)
        self.pos = torch.nn.Parameter(torch.zeros(1, self.max_ops + 2, H))
        torch.nn.init.normal_(self.task_embed, std=0.02)
        torch.nn.init.normal_(self.pos, std=0.02)
        enc = torch.nn.TransformerEncoderLayer(
            d_model=H, nhead=config.num_heads, dim_feedforward=4 * H, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            enc, num_layers=config.depth, norm=torch.nn.LayerNorm(H),
            enable_nested_tensor=False,
        )
        self.delta_head = torch.nn.Linear(H, self.nn2)
        self.stop_head = torch.nn.Linear(H, 1)

    def _teacher_sequence(self, a):
        """Ground-truth elimination: states A_0..A_m and operator deltas
        Delta_1..Delta_m (each I+Delta = elementary row op). Fixed column order."""
        n = self.n
        states = [a]
        deltas = []
        cur = a
        for k in range(n - 1):
            for i in range(k + 1, n):
                c = cur[:, i, k] / cur[:, k, k]  # (B,)
                delta = torch.zeros_like(cur)
                delta[:, i, k] = -c  # operator M = I + delta (single off-diag)
                cur = cur + delta @ cur  # apply M A
                states.append(cur)
                deltas.append(delta)
        return states, deltas

    def _run_transformer(self, toks):
        L = toks.shape[1]
        mask = torch.triu(
            torch.ones(L, L, device=toks.device, dtype=torch.bool), diagonal=1
        )
        return self.transformer(toks + self.pos[:, :L], mask=mask, is_causal=True)

    def forward(self, batch):
        a = batch["input"]
        B = a.shape[0]
        n = self.n
        states, deltas = self._teacher_sequence(a)  # m+1 states, m deltas
        m = len(deltas)
        state_emb = self.matrix_embed(
            torch.stack(states, dim=1).reshape(B, m + 1, self.nn2)
        )  # (B, m+1, H)
        toks = torch.cat([self.task_embed.expand(B, 1, -1), state_emb], dim=1)
        h = self._run_transformer(toks)  # (B, m+2, H)
        state_h = h[:, 1:, :]  # aligned with A_0..A_m  -> (B, m+1, H)

        delta_pred = self.delta_head(state_h).reshape(B, m + 1, n, n)
        stop_logit = self.stop_head(state_h).squeeze(-1)  # (B, m+1)

        delta_tgt = torch.stack(deltas, dim=1)  # (B, m, n, n)
        # smooth-L1 (robust to heavy-tailed multipliers from small no-pivot pivots)
        delta_loss = torch.nn.functional.smooth_l1_loss(
            delta_pred[:, :m], delta_tgt, reduction="none"
        ).sum(dim=(2, 3)).mean(dim=1)  # (B,)
        stop_tgt = torch.zeros(B, m + 1, device=a.device)
        stop_tgt[:, m] = 1.0
        stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            stop_logit, stop_tgt, reduction="none"
        ).mean(dim=1)  # (B,)

        diag = states[-1].diagonal(dim1=1, dim2=2)
        logdet = torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1)  # teacher
        return dict(
            logits=logdet.unsqueeze(-1),
            delta_loss=delta_loss,
            stop_loss=stop_loss,
            predictions=logdet.detach().unsqueeze(-1),
        )

    @torch.no_grad()
    def free_rollout(self, a, max_ops=None, stop_thresh=0.5):
        """Autoregressive: predict delta, apply M=I+delta, append state, until
        STOP or max_ops. Returns (logdet, n_ops, final_state)."""
        n = self.n
        if max_ops is None:
            max_ops = self.max_ops
        B = a.shape[0]
        states = [a]
        cur = a
        alive = torch.ones(B, dtype=torch.bool, device=a.device)
        n_ops = torch.zeros(B, device=a.device)
        for _ in range(max_ops):
            emb = self.matrix_embed(torch.stack(states, dim=1).reshape(B, len(states), self.nn2))
            toks = torch.cat([self.task_embed.expand(B, 1, -1), emb], dim=1)
            h = self._run_transformer(toks)
            last = h[:, -1, :]  # prediction at the most recent state
            stop = torch.sigmoid(self.stop_head(last).squeeze(-1)) > stop_thresh
            delta = self.delta_head(last).reshape(B, n, n)
            step = alive & (~stop)
            cur = torch.where(step.view(B, 1, 1), cur + delta @ cur, cur)
            n_ops = n_ops + step.float()
            alive = alive & (~stop)
            states.append(cur)
            if not alive.any():
                break
        diag = cur.diagonal(dim1=1, dim2=2)
        logdet = torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1)
        return logdet, n_ops, cur
