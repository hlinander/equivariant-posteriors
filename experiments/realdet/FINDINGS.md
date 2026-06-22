# Learning the determinant of real N×N matrices — findings

Investigation of *how much data / what it takes* to learn `det` of real Gaussian
matrices, and whether an autoregressive "operator rollout" can learn to *compute*
it rather than regress it. Three project tags in the analytics:
`real_det` (MLP regression), `real_det_elim` (elimination rollout),
`real_det_operator` (matrix-operator-token transformer).

Common setup: Gaussian-iid entries; target `(sign, log|det|)` via `slogdet`.
The sharp diagnostic throughout is **`jac_cos`** — cosine alignment of the
model's input→log|det| Jacobian with Jacobi's analytic gradient
`∂log|det A|/∂A = A^{-T}`. High `jac_cos` ⇒ the model learned the *function*;
high R² with low `jac_cos` ⇒ it learned a *value proxy*.

---

## 1. Direct MLP regression (`real_det`)

Target log|det| standardized to unit variance, so **R² = 1 − MSE**. Learning
curves over N and `n_train`; fixed-step budget (v2) to decouple data from compute.

**"How much data" has two different answers:**

| criterion | N=3 | N=4 | N=5 | N=6 | N=7 |
|---|---|---|---|---|---|
| **value** (val R² ≥ 0.9) | ~11k | ~22k | ~25k | ~25k | ~25k |
| **function** (jac_cos ≥ 0.9) | ~12k | ≫128k | unreached | unreached | unreached |

- **Predicting the value is easy and ~N-independent.** log|det| concentrates and
  is smoothly regressable from coarse statistics — **no factorial wall**. An MLP
  hits R²≈0.99 up to N=7 with ~25k samples.
- **Learning the function is where the wall lives.** At fixed data `jac_cos`
  collapses with N (N=3 0.97 → N=5 0.12 → N=7 0.07). At N≥5 you get the trap of
  **R²≈0.99 with jac_cos<0.15** — a model that "predicts the determinant" having
  learned almost nothing of the determinant function.
- N=4 deep-data (to 1M): `jac_cos` plateaus ~0.87 and val R² *falls* as the model
  is pushed toward the true sensitivity it lacks capacity to fit — capacity-limited,
  not just data-limited.

This motivated learning to *compute* det instead of regressing it.

---

## 2. Elimination rollout (`real_det_elim`)

Row operations as the "language": det-preserving elementary ops
`row_i -= c·row_k` (det=1 for any c), readout `Σlog|diag|`. A weight-tied step
module predicts the multiplier `c`; the rollout is unrolled and BPTT-trained.
A ladder of fixes, each removing one binding constraint:

- **log input** (multiplier is a ratio → subtraction): cracks the single division
  (N=2 MAE 0.26→0.04) but **fails N≥4** — past the first column entries become
  Schur complements (sums) where log isn't linear; log even *hurt* N=4.
- **log output** (head emits `log|c|`,sign): division at scale (N=3 mult_loss
  333→2.5, N=4 ~12k→12 for the multiplier targets).
- **teacher forcing / scheduled sampling**: fixes autoregressive drift (N=3 free
  MAE 1.18→0.41); confirmed classic exposure-bias.
- **pivoting** (none/partial/learned): the residual wall at N≥3 was *conditioning*
  — small pivots give heavy-tailed `c`. Partial pivoting bounds `|c|≤1` and
  roughly halves error; **learned pivoting matched partial** (the model
  *discovered* the pivot strategy — every pivot choice = the argmax).
- **test-time refinement** (greedy extra eliminations): **3× error reduction at
  N=3**, but **flat at N≥4** — refinement only contracts if per-step multipliers
  are accurate enough; below that it's a fixed point.

**Two binding constraints surfaced:** (a) **per-step multiplier precision**
floored at N=4 — MLP step ~0.19, transformer step ~0.15, both worsening with N;
(b) the **per-step-re-encoded, deep-BPTT unrolled rollout doesn't optimize at
depth** — the transformer step needed lr=1e-4 + a final LayerNorm just to train,
and still plateaued at N=4. Both point away from "unroll + BPTT".

---

## 3. Matrix-operator-token transformer (`real_det_operator`)

Reframed as a **sequence model over an operator algebra**: tokens are matrices,
the model autoregressively predicts the **next operator as a full matrix M**
(`A ← M·A`; head initialized to ~identity for a no-op default) plus a discrete
**STOP**. Teacher = partial-pivoting GE as full-matrix operators (bounded
entries). Trained as a **parallel teacher-forced causal LM** — no per-step
re-encoding, no deep BPTT. This is the structural fix.

**Breaks the N≥4 wall.** op_loss ≈ **0.003, N-independent** (vs the unrolled
MLP/transformer steps that floored at 0.15–0.19 *at N=4*):

| N | op_loss | free_logdet_mae | free_avg_ops (true) | jac_cos |
|---|---|---|---|---|
| 3 | 0.003 | 0.12 | 4.00 (4) | 0.91 |
| 4 | 0.004 | 0.19 | 6.00 (6) | 0.84 |
| 5 | 0.003 | 0.26 | 8.00 (8) | 0.83 |
| 8 | 0.0025 | 0.38 | 13.99 (14) | ~0.73 |

- **Learns the true function** — `jac_cos` 0.83–0.91 and **degrades gracefully**
  with N, where the MLP regression *collapsed* to 0.14 at N=5. Clearest evidence
  that "learning to compute" beats "learning the function".
- **N=8 cracked** — a size hopeless for direct regression: jac_cos ~0.73,
  exactly 14 operators recovered, correct halting. Traces (inspect_operator.py)
  show it genuinely pivots + eliminates column-by-column then STOPs.
- **Data-limited at N=4/5** (more data → lower op_loss/free_mae, higher jac_cos);
  **optimization-limited at N=8** (the 10×-data run was *worse* because epochs
  were cut — undertrained, not data-starved).

**Test-time inference study.** STOP halts exactly at the true op count; forced
extra compute = near-identity no-ops (`‖M−I‖→~0.005`, error flat) — overthinking
is provably safe (det-conservation, learned). Per-operator error drops
monotonically. **But no compute *scaling*** — fixed-order GE is fixed-compute
(`2(n−1)` ops, instance-independent), so "think more" can't help here.

**Emergent thinking (softened teacher forcing) — negative result.** Annealing
per-step operator supervision to 0 while learning from an end-state objective
(triangularize + match log|det| on the model's own rollout): the model
**triangularizes but does not preserve the determinant** (tri_loss low ~0.2,
read_loss stuck ~1.3, jac_cos 0.27). It finds a det-non-preserving local minimum
the end-state gradient can't escape — **imitation is load-bearing**; discovering
the algorithm from sparse outcome signal is the hard, unsolved part (the
credit-assignment wall, mirroring why LLMs need structure to learn to reason).

---

## Headline takeaways

1. Over ℝ with the (sign, log|det|) target, **predicting the value of det is
   easy and N-independent** — the factorial wall is about *coefficient recovery*,
   not value regression.
2. **Learning the determinant *function*** (correct `A^{-T}` sensitivity) is the
   hard part, and direct regression hits a wall (jac_cos collapses by N=5).
3. An **autoregressive operator-rollout transformer learns to *compute* det**,
   matching the true function gracefully up to **N=8**, where regression fails —
   provided operators are **supervised** (teacher-forced); the parallel-causal-LM
   formulation is what made it scale.
4. The model has the right **inference-time properties** (correct halting, safe
   overthinking) but the task is fixed-compute, so it shows no test-time scaling.
5. **Emergent discovery from outcome reward fails** here — the remaining open
   problem, and the natural bridge to RL / partial-supervision approaches.

## Tooling
- `inspect_operator.py` — step-by-step rollout traces (operators, residual, STOP).
- `ftle_study.py` — `jac_cos`/relerr vs Jacobi `A^{-T}`.
- `inference_study.py` — compute-vs-accuracy, halting, identity-padding.
- `scaling_curve.py` — post-hoc refinement R-sweep (elimination model).
- `operator_rollout_design.md` — the general-substrate design notes.

## Open directions
- Cross-N generalization (train N≤4, test N=6,7) — needs size-agnostic
  (entry-token / relative-position, permutation-equivariant) encoding.
- True test-time compute scaling — refinement *operators* in the slack region,
  or variable-length algorithms.
- Emergent operators — partial-anneal / up-weighted readout / RL on the
  verifiable det reward.
- Multitask "sentences" — `[INV]` (target=identity → composed op = A⁻¹), `[SOLVE]`,
  etc., sharing the operator vocabulary (operator-algebra calculator).
