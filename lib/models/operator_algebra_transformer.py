"""Operator-algebra transformer: one causal sequence model over a shared token
space, where each algorithm is a "sentence".

Shared token space (all embedded to hidden H, + a type embedding + position):
    task token   : which sentence (DET / INV / ICL)
    matrix token : an n×n matrix (a rollout state, or an emitted operator)
    vector token : an n-vector (ICL context x/y, or the query)
Shared heads: op_head -> n×n operator matrix; stop_head -> halt logit.

Sentences:
    DET : [DET][A_0][A_1]...  multiplicative operator rollout (A<-M A) to upper
          triangular; readout Sum log|diag|. Teacher = partial-pivoting GE ops.
    INV : like DET but Gauss-Jordan to identity; composed operators = A^{-1}.
    ICL : [ICL][x_1][y_1]...[x_k][y_k][q] then emit rank-1 partial operators that
          ADD up to the latent W (W x_i = y_i inferred in-context); apply to q.
          Teacher = SVD components of W in decreasing singular order (Eckart-
          Young => more operators monotonically better = test-time compute).

Two operator semantics coexist: DET/INV multiply (no-op = I, state fed back),
ICL adds (no-op = 0, emitted operator fed back). The composition is per-sentence;
the token space, transformer and heads are shared.
"""
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec

TASK_DET, TASK_INV, TASK_ICL = 0, 1, 2
TT_TASK, TT_MSTATE, TT_OP, TT_VX, TT_VY, TT_QUERY = 0, 1, 2, 3, 4, 5


@dataclass(frozen=True)
class OperatorAlgebraConfig:
    hidden: int = 256
    depth: int = 4
    num_heads: int = 8
    eps: float = 1e-6
    slack: int = 2
    svd_thresh: float = 1e-3  # ICL: STOP once singular value below this

    def serialize_human(self):
        return self.__dict__


class OperatorAlgebraTransformer(torch.nn.Module):
    def __init__(self, config: OperatorAlgebraConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.n = n = data_spec.input_shape[0]
        H = config.hidden
        self.nn2 = n * n
        self.true_det = 2 * (n - 1)
        self.true_inv = 3 * n
        self.max_ops = max(self.true_det, self.true_inv, n) + config.slack
        self.max_len = 2 + 2 * self.max_ops + 1 + self.max_ops  # generous

        self.task_embed = torch.nn.Embedding(3, H)
        self.type_embed = torch.nn.Embedding(6, H)
        self.matrix_embed = torch.nn.Linear(self.nn2, H)
        self.vector_embed = torch.nn.Linear(n, H)
        self.pos = torch.nn.Parameter(torch.zeros(1, self.max_len, H))
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

    # ---- embedding helpers ----
    def _mat_tok(self, m, type_id):  # m: (B,*,n,n) -> (B,*,H)
        return self.matrix_embed(m.reshape(*m.shape[:-2], self.nn2)) + self.type_embed(
            torch.tensor(type_id, device=m.device)
        )

    def _vec_tok(self, v, type_id):  # v: (B,*,n) -> (B,*,H)
        return self.vector_embed(v) + self.type_embed(torch.tensor(type_id, device=v.device))

    def _run(self, toks):
        L = toks.shape[1]
        toks = toks + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, device=toks.device, dtype=torch.bool), diagonal=1)
        return self.transformer(toks, mask=mask, is_causal=True)

    def _eye(self, B, device, dtype):
        return torch.eye(self.n, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    # ================= ICL sentence =================
    def _icl_teacher(self, W):
        """SVD components M_t = s_t u_t v_t^T (decreasing), padded to max_ops with
        zero ops; STOP once s_t < thresh."""
        B, d = W.shape[0], self.n
        U, S, Vh = torch.linalg.svd(W)
        # rank-1 SVD components M_t = s_t u_t v_t^T, decreasing singular order
        comps = []
        for t in range(d):
            comps.append(S[:, t].view(B, 1, 1) * torch.einsum("bi,bj->bij", U[:, :, t], Vh[:, t, :]))
        ops = torch.stack(comps, dim=1)  # (B, d, n, n)
        T = self.max_ops
        if d < T:
            pad = torch.zeros(B, T - d, self.n, self.n, device=W.device, dtype=W.dtype)
            ops = torch.cat([ops, pad], dim=1)
        eff_rank = (S > self.config.svd_thresh).sum(dim=1)  # (B,)
        return ops, eff_rank  # ops (B,T,n,n)

    def _icl_forward(self, batch):
        cx, cy = batch["context_x"], batch["context_y"]  # (B,k,n)
        qx, W = batch["query_x"], batch["W"]
        B, k, n = cx.shape
        device = cx.device
        ops, eff_rank = self._icl_teacher(W)  # (B,T,n,n)
        T = ops.shape[1]

        # tokens: [TASK] x1 y1 ... xk yk [Q] op1 ... op_{T-1}
        toks = [self.task_embed(torch.full((B,), TASK_ICL, device=device)).unsqueeze(1)]
        for i in range(k):
            toks.append(self._vec_tok(cx[:, i], TT_VX).unsqueeze(1))
            toks.append(self._vec_tok(cy[:, i], TT_VY).unsqueeze(1))
        toks.append(self._vec_tok(qx, TT_QUERY).unsqueeze(1))
        emit_start = len(toks)  # index of the query token's position output -> predicts op1
        for t in range(T - 1):
            toks.append(self._mat_tok(ops[:, t], TT_OP).unsqueeze(1))
        seq = torch.cat(toks, dim=1)  # (B, L, H)
        h = self._run(seq)
        # the query token is at index emit_start-1; its output predicts op1; then op tokens
        pred_h = h[:, emit_start - 1 : emit_start - 1 + T, :]  # (B,T,H)
        op_pred = self.op_head(pred_h).reshape(B, T, n, n)
        op_loss = torch.nn.functional.smooth_l1_loss(op_pred, ops, reduction="none").sum(dim=(2, 3)).mean(dim=1)
        stop_logit = self.stop_head(pred_h).squeeze(-1)  # (B,T)
        idx = torch.arange(T, device=device).unsqueeze(0)
        stop_tgt = (idx >= eff_rank.unsqueeze(1)).float()
        stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            stop_logit, stop_tgt, reduction="none"
        ).mean(dim=1)
        # teacher readout (metric): full-rank reconstruction of W q
        What = ops.sum(dim=1)
        yhat = torch.einsum("bij,bj->bi", What, qx)
        return dict(
            logits=yhat, loss=op_loss + stop_loss, op_loss=op_loss, stop_loss=stop_loss,
            predictions=yhat.detach(),
        )

    @torch.no_grad()
    def icl_rollout(self, batch, k=None, max_ops=None, stop_thresh=0.5):
        """Free rollout: infer + emit rank-1 ops from context, accumulate W_hat,
        apply to query. Returns (yhat, n_ops, per_step_yhat list). k truncates
        the context (for the ICL-vs-examples curve)."""
        cx, cy, qx = batch["context_x"], batch["context_y"], batch["query_x"]
        B, kk, n = cx.shape
        if k is None:
            k = kk
        if max_ops is None:
            max_ops = self.max_ops
        device = cx.device
        base = [self.task_embed(torch.full((B,), TASK_ICL, device=device)).unsqueeze(1)]
        for i in range(k):
            base.append(self._vec_tok(cx[:, i], TT_VX).unsqueeze(1))
            base.append(self._vec_tok(cy[:, i], TT_VY).unsqueeze(1))
        base.append(self._vec_tok(qx, TT_QUERY).unsqueeze(1))
        seq = torch.cat(base, dim=1)
        What = torch.zeros(B, n, n, device=device)
        alive = torch.ones(B, dtype=torch.bool, device=device)
        n_ops = torch.zeros(B, device=device)
        per_step = []
        cur = seq
        for t in range(max_ops):
            h = self._run(cur)[:, -1, :]
            stop = torch.sigmoid(self.stop_head(h).squeeze(-1)) > stop_thresh
            M = self.op_head(h).reshape(B, n, n)
            step = (alive & (~stop)).view(B, 1, 1)
            What = What + torch.where(step, M, torch.zeros_like(M))
            n_ops = n_ops + (alive & (~stop)).float()
            alive = alive & (~stop)
            per_step.append(torch.einsum("bij,bj->bi", What, qx))
            cur = torch.cat([cur, self._mat_tok(M, TT_OP).unsqueeze(1)], dim=1)
            if not alive.any():
                break
        yhat = torch.einsum("bij,bj->bi", What, qx)
        return yhat, n_ops, per_step

    # ================= DET / INV sentences (matrix rollout) =================
    def _perm(self, cur, k, eye, bidx):
        p = cur[:, k:, k].abs().argmax(dim=1) + k
        rows = torch.arange(self.n, device=cur.device).unsqueeze(0).expand(cur.shape[0], -1).clone()
        rows[bidx, k] = p
        rows[bidx, p] = k
        return torch.gather(eye, 1, rows.unsqueeze(-1).expand(-1, -1, self.n))

    def _matrix_teacher(self, a, task):
        n, B = self.n, a.shape[0]
        bidx = torch.arange(B, device=a.device)
        eye = self._eye(B, a.device, a.dtype)
        states, ops = [a], []
        cur = a
        if task == TASK_DET:
            for k in range(n - 1):
                P = self._perm(cur, k, eye, bidx); cur = P @ cur; states.append(cur); ops.append(P)
                E = eye.clone()
                c = cur[:, k + 1 :, k] / cur[:, k, k].unsqueeze(1)
                E[bidx.unsqueeze(1), torch.arange(k + 1, n, device=a.device).unsqueeze(0), k] = -c
                cur = E @ cur; states.append(cur); ops.append(E)
            true_len = self.true_det
        else:  # INV
            for k in range(n):
                P = self._perm(cur, k, eye, bidx); cur = P @ cur; states.append(cur); ops.append(P)
                S = eye.clone(); S[bidx, k, k] = 1.0 / cur[:, k, k]; cur = S @ cur; states.append(cur); ops.append(S)
                E = eye.clone(); col = cur[:, :, k].clone(); E[:, :, k] = -col; E[bidx, k, k] = 1.0
                cur = E @ cur; states.append(cur); ops.append(E)
            true_len = self.true_inv
        L = self.max_ops
        last = states[-1]
        while len(ops) < L:
            ops.append(eye); states.append(last)
        return torch.stack(states, 1), torch.stack(ops, 1), true_len

    def _matrix_forward(self, a, task_id):
        n, B = self.n, a.shape[0]
        states, ops, true_len = self._matrix_teacher(a, task_id)
        L = ops.shape[1]
        task_tok = self.task_embed(torch.full((B,), task_id, device=a.device)).unsqueeze(1)
        state_tok = self._mat_tok(states, TT_MSTATE)  # (B,L+1,H)
        seq = torch.cat([task_tok, state_tok], dim=1)
        h = self._run(seq)[:, 1:, :]
        op_pred = self.op_head(h).reshape(B, L + 1, n, n)
        op_loss = torch.nn.functional.smooth_l1_loss(op_pred[:, :L], ops, reduction="none").sum(dim=(2, 3)).mean(dim=1)
        stop_logit = self.stop_head(h).squeeze(-1)
        idx = torch.arange(L + 1, device=a.device).unsqueeze(0)
        stop_tgt = (idx >= true_len).float().expand(B, -1)
        stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(stop_logit, stop_tgt, reduction="none").mean(dim=1)
        diag = states[:, self.true_det].diagonal(dim1=1, dim2=2)
        logdet = torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1)
        return dict(logits=logdet.unsqueeze(-1), loss=op_loss + stop_loss,
                    op_loss=op_loss, stop_loss=stop_loss, predictions=logdet.detach().unsqueeze(-1))

    def forward(self, batch):
        if self.training:
            self.train_steps += 1
        if "context_x" in batch:
            return self._icl_forward(batch)
        task_id = int(batch["task"][0].item()) if "task" in batch else TASK_DET
        return self._matrix_forward(batch["input"], task_id)
