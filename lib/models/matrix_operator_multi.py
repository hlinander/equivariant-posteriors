"""Multitask matrix-operator-token transformer: one model over an operator
algebra, trained on several "sentences" sharing the operator vocabulary.

[TASK] [A_0] [A_1] ... causal sequence; at each state a full operator matrix M
is predicted (A <- M A) plus STOP. A per-task token selects the task:
    DET (0): partial-pivoting GE -> upper-triangular; readout Sum log|diag|.
    INV (1): Gauss-Jordan -> identity; the composed operator product is A^{-1}.

Tasks have different true lengths (DET 2(n-1), INV 3n); both teacher sequences
are built, identity-padded to a common length, and selected per sample by task.
Trained teacher-forced as a parallel causal LM (op smooth-L1 + STOP BCE).
"""
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass(frozen=True)
class MatrixOperatorMultiConfig:
    hidden: int = 256
    depth: int = 4
    num_heads: int = 8
    eps: float = 1e-6
    slack: int = 4
    num_tasks: int = 2

    def serialize_human(self):
        return self.__dict__


class MatrixOperatorMulti(torch.nn.Module):
    def __init__(self, config: MatrixOperatorMultiConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.n = n = data_spec.input_shape[0]
        H = config.hidden
        self.nn2 = n * n
        self.true_det = 2 * (n - 1)
        self.true_inv = 3 * n
        self.max_ops = max(self.true_det, self.true_inv) + config.slack

        self.matrix_embed = torch.nn.Linear(self.nn2, H)
        self.task_embed = torch.nn.Embedding(config.num_tasks, H)
        self.pos = torch.nn.Parameter(torch.zeros(1, self.max_ops + 2, H))
        torch.nn.init.normal_(self.pos, std=0.02)
        enc = torch.nn.TransformerEncoderLayer(
            d_model=H, nhead=config.num_heads, dim_feedforward=4 * H, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            enc, num_layers=config.depth, norm=torch.nn.LayerNorm(H),
            enable_nested_tensor=False,
        )
        self.op_head = torch.nn.Linear(H, self.nn2)
        torch.nn.init.zeros_(self.op_head.weight)
        self.op_head.bias.data = torch.eye(n).reshape(-1).clone()
        self.stop_head = torch.nn.Linear(H, 1)
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long))

    def _eye(self, B, device, dtype):
        return torch.eye(self.n, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    def _perm(self, cur, k, eye, bidx):
        p = cur[:, k:, k].abs().argmax(dim=1) + k
        rows = torch.arange(self.n, device=cur.device).unsqueeze(0).expand(cur.shape[0], -1).clone()
        rows[bidx, k] = p
        rows[bidx, p] = k
        return torch.gather(eye, 1, rows.unsqueeze(-1).expand(-1, -1, self.n))

    def _teacher_det(self, a):
        n = self.n
        B = a.shape[0]
        bidx = torch.arange(B, device=a.device)
        eye = self._eye(B, a.device, a.dtype)
        states = [a]
        ops = []
        cur = a
        for k in range(n - 1):
            P = self._perm(cur, k, eye, bidx)
            cur = P @ cur
            states.append(cur)
            ops.append(P)
            E = eye.clone()
            c = cur[:, k + 1 :, k] / cur[:, k, k].unsqueeze(1)
            E[bidx.unsqueeze(1), torch.arange(k + 1, n, device=a.device).unsqueeze(0), k] = -c
            cur = E @ cur
            states.append(cur)
            ops.append(E)
        return states, ops  # 2(n-1) ops

    def _teacher_inv(self, a):
        n = self.n
        B = a.shape[0]
        bidx = torch.arange(B, device=a.device)
        eye = self._eye(B, a.device, a.dtype)
        states = [a]
        ops = []
        cur = a
        for k in range(n):
            P = self._perm(cur, k, eye, bidx)
            cur = P @ cur
            states.append(cur)
            ops.append(P)
            # scale row k to unit pivot
            S = eye.clone()
            S[bidx, k, k] = 1.0 / cur[:, k, k]
            cur = S @ cur
            states.append(cur)
            ops.append(S)
            # eliminate all other rows in column k
            E = eye.clone()
            col = cur[:, :, k].clone()
            E[:, :, k] = -col
            E[bidx, k, k] = 1.0
            cur = E @ cur
            states.append(cur)
            ops.append(E)
        return states, ops  # 3n ops

    def _pad(self, states, ops, L, eye):
        """Pad to L ops (L+1 states) with identity ops + repeated final state."""
        states = list(states)
        ops = list(ops)
        last = states[-1]
        while len(ops) < L:
            ops.append(eye)
            states.append(last)
        return torch.stack(states, dim=1), torch.stack(ops, dim=1)  # (B,L+1,n,n),(B,L,n,n)

    def _build_teacher(self, a, task):
        B = a.shape[0]
        eye = self._eye(B, a.device, a.dtype)
        L = self.max_ops
        ds, do = self._pad(*self._teacher_det(a), L, eye)
        is_, io = self._pad(*self._teacher_inv(a), L, eye)
        sel = (task == 0).view(B, 1, 1, 1)
        states = torch.where(sel, ds, is_)
        ops = torch.where(sel, do, io)
        true_len = torch.where(task == 0, torch.full_like(task, self.true_det),
                               torch.full_like(task, self.true_inv))
        return states, ops, true_len

    def _run(self, toks):
        Lq = toks.shape[1]
        mask = torch.triu(torch.ones(Lq, Lq, device=toks.device, dtype=torch.bool), diagonal=1)
        return self.transformer(toks + self.pos[:, :Lq], mask=mask, is_causal=True)

    def forward(self, batch):
        a = batch["input"]
        task = batch["task"].long()
        B, n = a.shape[0], self.n
        states, ops, true_len = self._build_teacher(a, task)  # (B,L+1,n,n),(B,L,n,n)
        L = ops.shape[1]
        emb = self.matrix_embed(states.reshape(B, L + 1, self.nn2))
        toks = torch.cat([self.task_embed(task).unsqueeze(1), emb], dim=1)
        h = self._run(toks)[:, 1:, :]  # aligned with A_0..A_L
        op_pred = self.op_head(h).reshape(B, L + 1, n, n)
        stop_logit = self.stop_head(h).squeeze(-1)  # (B, L+1)

        op_loss = torch.nn.functional.smooth_l1_loss(
            op_pred[:, :L], ops, reduction="none"
        ).sum(dim=(2, 3)).mean(dim=1)
        idx = torch.arange(L + 1, device=a.device).unsqueeze(0)
        stop_tgt = (idx >= true_len.unsqueeze(1)).float()
        stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            stop_logit, stop_tgt, reduction="none"
        ).mean(dim=1)
        if self.training:
            self.train_steps += 1
        diag = states[:, self.true_det].diagonal(dim1=1, dim2=2)  # det readout slot
        logdet = torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1)
        return dict(
            logits=logdet.unsqueeze(-1),
            loss=op_loss + stop_loss,
            op_loss=op_loss,
            stop_loss=stop_loss,
            predictions=logdet.detach().unsqueeze(-1),
        )

    @torch.no_grad()
    def free_rollout(self, a, task, stop_thresh=0.5):
        """Per-sample free rollout. Returns (final_state, op_product P, n_ops).
        P satisfies P @ A = final; for INV final≈I so P≈A^{-1}."""
        n, B = self.n, a.shape[0]
        task = task.long()
        states = [a]
        cur = a
        P = self._eye(B, a.device, a.dtype)
        alive = torch.ones(B, dtype=torch.bool, device=a.device)
        n_ops = torch.zeros(B, device=a.device)
        for _ in range(self.max_ops):
            emb = self.matrix_embed(torch.stack(states, dim=1).reshape(B, len(states), self.nn2))
            toks = torch.cat([self.task_embed(task).unsqueeze(1), emb], dim=1)
            last = self._run(toks)[:, -1, :]
            stop = torch.sigmoid(self.stop_head(last).squeeze(-1)) > stop_thresh
            M = self.op_head(last).reshape(B, n, n)
            step = (alive & (~stop)).view(B, 1, 1)
            cur = torch.where(step, M @ cur, cur)
            P = torch.where(step, M @ P, P)
            n_ops = n_ops + step.view(B).float()
            alive = alive & (~stop)
            states.append(cur)
            if not alive.any():
                break
        return cur, P, n_ops
