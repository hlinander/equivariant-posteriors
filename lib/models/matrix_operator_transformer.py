"""Matrix-operator-token transformer: a sequence model over an operator algebra.

Tokens are full matrices. The sequence is [TASK] [A_0] [A_1] ... [A_m], where
each A_t is the matrix state after t predicted operators. At each state position
a causal transformer predicts the NEXT operator as a **full matrix** M (applied
A <- M A) plus a discrete STOP. Every operation is just an operator matrix --
permutations (row swaps), eliminations, identity (no-op) are all the same kind
of object, predicted uniformly. The head is initialized to output ~identity so
the default action is a no-op (trainability without an I+Delta restriction).

Single-task DET: reduce A to upper-triangular by a sequence of operators, read
Sum log|diag|. The teacher is partial-pivoting Gaussian elimination, whose
operators (permutation P_k, column-elimination E_k) are full matrices with
bounded entries (|multiplier| <= 1). Trained as a parallel teacher-forced causal
sequence model (no per-step re-encoding / deep BPTT).
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
    max_ops: int = 0  # 0 -> 2*(n-1) + slack
    slack: int = 0  # extra identity-operator slots past the elimination, so STOP
    # actually halts (not the max_ops cap) and "think more" = emit identities
    # Emergent mode: soften per-step operator supervision (op_loss) and learn
    # the operators from an end-state objective on the model's OWN free rollout
    # (triangularize + match the log|det| readout). op_loss weight anneals
    # 1 -> 0 over op_anneal_steps; if it holds, the model internalized the ops.
    endstate: bool = False
    op_anneal_steps: int = 0
    tri_weight: float = 1.0
    read_weight: float = 1.0

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
        self.true_ops = 2 * (self.n - 1)
        self.max_ops = config.max_ops or (self.true_ops + config.slack)
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
        # operator head: predict a full n x n operator matrix, initialized to ~I
        self.op_head = torch.nn.Linear(H, self.nn2)
        torch.nn.init.zeros_(self.op_head.weight)
        self.op_head.bias.data = torch.eye(self.n).reshape(-1).clone()
        self.stop_head = torch.nn.Linear(H, 1)
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long))

    def _teacher_sequence(self, a):
        """Partial-pivoting GE as full-matrix operators. Per column k: a
        permutation P_k (swap largest-|.| pivot up; identity if none) then a
        column-elimination E_k. Returns states A_0..A_m and operators M_1..M_m
        (each (B,n,n)); |elimination multipliers| <= 1."""
        n = self.n
        B = a.shape[0]
        bidx = torch.arange(B, device=a.device)
        eye = torch.eye(n, device=a.device).unsqueeze(0).expand(B, -1, -1)
        states = [a]
        ops = []
        cur = a
        for k in range(n - 1):
            # permutation operator (partial pivot)
            p = cur[:, k:, k].abs().argmax(dim=1) + k  # (B,)
            P = eye.clone()
            # swap rows k and p of the identity, per sample
            rows = torch.arange(n, device=a.device).unsqueeze(0).expand(B, -1).clone()
            rows[bidx, k] = p
            rows[bidx, p] = k
            P = eye[bidx][:, :, :]  # (B,n,n) identity
            P = torch.gather(P, 1, rows.unsqueeze(-1).expand(-1, -1, n))
            cur = P @ cur
            states.append(cur)
            ops.append(P)
            # column-elimination operator
            E = eye.clone()
            c = cur[:, k + 1 :, k] / cur[:, k, k].unsqueeze(1)  # (B, n-k-1)
            E[bidx.unsqueeze(1), torch.arange(k + 1, n, device=a.device).unsqueeze(0), k] = -c
            cur = E @ cur
            states.append(cur)
            ops.append(E)
        # identity-operator padding past the elimination (think-more = no-op)
        for _ in range(self.config.slack):
            ops.append(eye[bidx])
            states.append(cur)
        return states, ops

    def _run_transformer(self, toks):
        L = toks.shape[1]
        mask = torch.triu(torch.ones(L, L, device=toks.device, dtype=torch.bool), diagonal=1)
        return self.transformer(toks + self.pos[:, :L], mask=mask, is_causal=True)

    def forward(self, batch):
        a = batch["input"]
        B = a.shape[0]
        n = self.n
        states, ops = self._teacher_sequence(a)
        m = len(ops)
        state_emb = self.matrix_embed(torch.stack(states, dim=1).reshape(B, m + 1, self.nn2))
        toks = torch.cat([self.task_embed.expand(B, 1, -1), state_emb], dim=1)
        h = self._run_transformer(toks)
        state_h = h[:, 1:, :]  # aligned with A_0..A_m

        op_pred = self.op_head(state_h).reshape(B, m + 1, n, n)
        stop_logit = self.stop_head(state_h).squeeze(-1)

        op_tgt = torch.stack(ops, dim=1)  # (B, m, n, n); ops past true_ops are I
        op_loss = torch.nn.functional.smooth_l1_loss(
            op_pred[:, :m], op_tgt, reduction="none"
        ).sum(dim=(2, 3)).mean(dim=1)  # (B,)
        # STOP once the elimination is done (state index >= true_ops), so STOP
        # halting is learned and identity-padding fills any forced extra steps.
        stop_tgt = torch.zeros(B, m + 1, device=a.device)
        stop_tgt[:, self.true_ops :] = 1.0
        stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            stop_logit, stop_tgt, reduction="none"
        ).mean(dim=1)

        diag = states[-1].diagonal(dim1=1, dim2=2)
        logdet = torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1)

        op_w = 1.0
        if self.config.op_anneal_steps > 0 and self.training:
            op_w = max(0.0, 1.0 - float(self.train_steps.item()) / self.config.op_anneal_steps)
        combined = op_w * op_loss + stop_loss
        tri_loss = a.new_zeros(B)
        read_loss = a.new_zeros(B)
        if self.config.endstate:
            final = self._diff_rollout_final(a, self.true_ops)
            tri_loss = torch.tril(final, diagonal=-1).pow(2).sum(dim=(1, 2))
            df = final.diagonal(dim1=1, dim2=2)
            logdet_free = torch.log(df.abs().clamp_min(self.config.eps)).sum(dim=1)
            _, logdet_true = torch.linalg.slogdet(a)
            read_loss = (logdet_free - logdet_true).abs()
            combined = combined + self.config.tri_weight * tri_loss + self.config.read_weight * read_loss

        if self.training:
            self.train_steps += 1
        return dict(
            logits=logdet.unsqueeze(-1),
            loss=combined,
            op_loss=op_loss,
            stop_loss=stop_loss,
            tri_loss=tri_loss,
            read_loss=read_loss,
            predictions=logdet.detach().unsqueeze(-1),
        )

    def _diff_rollout_final(self, a, n_steps):
        """Grad-enabled free rollout (model's own operators) returning the final
        matrix state -- for the emergent end-state objective."""
        states = [a]
        cur = a
        for _ in range(n_steps):
            emb = self.matrix_embed(
                torch.stack(states, dim=1).reshape(a.shape[0], len(states), self.nn2)
            )
            toks = torch.cat([self.task_embed.expand(a.shape[0], 1, -1), emb], dim=1)
            last = self._run_transformer(toks)[:, -1, :]
            M = self.op_head(last).reshape(a.shape[0], self.n, self.n)
            cur = M @ cur
            states.append(cur)
        return cur

    def rollout_logdet(self, a, n_steps):
        """Grad-enabled free rollout returning only log|det| (B,), for the
        Jacobian / FTLE study (d log|det| / dA vs Jacobi's A^{-T}). Fixed n_steps
        (no STOP), fully differentiable (operators predicted as matrices)."""
        states = [a]
        cur = a
        for _ in range(n_steps):
            emb = self.matrix_embed(
                torch.stack(states, dim=1).reshape(a.shape[0], len(states), self.nn2)
            )
            toks = torch.cat([self.task_embed.expand(a.shape[0], 1, -1), emb], dim=1)
            last = self._run_transformer(toks)[:, -1, :]
            M = self.op_head(last).reshape(a.shape[0], self.n, self.n)
            cur = M @ cur
            states.append(cur)
        diag = cur.diagonal(dim1=1, dim2=2)
        return torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1)

    @torch.no_grad()
    def trace_rollout(self, a):
        """Single-example (batch=1) free rollout for inspection. Returns the
        initial matrix, per-step records (operator M, ||M-I||, stop_prob, the
        new state, its strict-lower residual norm), the final state, and the
        predicted log|det|. Halts on STOP (or max_ops)."""
        assert a.shape[0] == 1
        n = self.n
        states = [a]
        cur = a
        initial = a.clone()
        steps = []
        for t in range(self.max_ops):
            emb = self.matrix_embed(torch.stack(states, dim=1).reshape(1, len(states), self.nn2))
            toks = torch.cat([self.task_embed.expand(1, 1, -1), emb], dim=1)
            last = self._run_transformer(toks)[:, -1, :]
            stop_prob = torch.sigmoid(self.stop_head(last).squeeze(-1)).item()
            M = self.op_head(last).reshape(1, n, n)
            op_norm = (M[0] - torch.eye(n, device=a.device)).norm().item()
            if stop_prob > 0.5:
                steps.append(dict(stopped=True, stop_prob=stop_prob, op_norm=op_norm))
                break
            cur = M @ cur
            states.append(cur)
            steps.append(dict(
                stopped=False, stop_prob=stop_prob, op_norm=op_norm, M=M[0].clone(),
                matrix=cur[0].clone(),
                lower_resid=float(torch.tril(cur[0], diagonal=-1).norm()),
            ))
        diag = cur.diagonal(dim1=1, dim2=2)
        pred = float(torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1))
        return dict(initial=initial[0], steps=steps, final=cur[0], pred_logdet=pred)

    @torch.no_grad()
    def free_rollout_trace(self, a, n_steps):
        """Force exactly n_steps operators (ignore STOP). Per step records mean
        stop-probability, mean operator magnitude ||M-I||, and running log|det|
        -- for the test-time-compute study (does extra compute help? when does
        it want to halt? does it identity-pad past the natural stop?)."""
        n = self.n
        B = a.shape[0]
        eye = torch.eye(n, device=a.device)
        states = [a]
        cur = a
        stop_probs, op_norms, logdets = [], [], []
        for _ in range(n_steps):
            emb = self.matrix_embed(torch.stack(states, dim=1).reshape(B, len(states), self.nn2))
            toks = torch.cat([self.task_embed.expand(B, 1, -1), emb], dim=1)
            last = self._run_transformer(toks)[:, -1, :]
            stop_probs.append(torch.sigmoid(self.stop_head(last).squeeze(-1)).mean().item())
            M = self.op_head(last).reshape(B, n, n)
            op_norms.append((M - eye).norm(dim=(1, 2)).mean().item())
            cur = M @ cur
            states.append(cur)
            diag = cur.diagonal(dim1=1, dim2=2)
            logdets.append(torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1))
        return stop_probs, op_norms, logdets

    @torch.no_grad()
    def free_rollout(self, a, max_ops=None, stop_thresh=0.5):
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
            last = h[:, -1, :]
            stop = torch.sigmoid(self.stop_head(last).squeeze(-1)) > stop_thresh
            M = self.op_head(last).reshape(B, n, n)
            step = alive & (~stop)
            cur = torch.where(step.view(B, 1, 1), M @ cur, cur)
            n_ops = n_ops + step.float()
            alive = alive & (~stop)
            states.append(cur)
            if not alive.any():
                break
        diag = cur.diagonal(dim1=1, dim2=2)
        logdet = torch.log(diag.abs().clamp_min(self.config.eps)).sum(dim=1)
        return logdet, n_ops, cur
