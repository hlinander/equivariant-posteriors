# Operator-rollout transformer — design spec

Goal: a *general* model that predicts a transformation of a structured object by
an autoregressive rollout of elementary operators, supervised on the resulting
state — not a determinant solver. The real-determinant elimination work was the
verifiable testbed that validated the core machinery; this spec keeps that core
and drops the determinant-specific scaffolding.

## Keep (validated, general)
- Autoregressive **operator rollout**: encode state -> predict operator -> apply
  / compose -> repeat.
- **Log-space multiplicative output** head (ratios/scales -> additive, trainable).
- **Teacher forcing / scheduled sampling** for the rollout (exposure bias).
- **Test-time compute**: keep applying operators until an end-state objective is
  met — now *intrinsic*, not a bolted-on refinement phase.
- Transformer step trained with a transformer recipe (low LR + final LayerNorm;
  no warmup available in-framework, so lean on LR).

## Drop (determinant-specific / ad hoc)
- Pivoting as a hand-coded policy (let attention decide what to operate on).
- The `sum log|diag|` triangular readout and the lower-tri penalty.
- The `det=1` conservation trick; elementary-row-op-only operator family.

## Encoder: entry-token transformer (general substrate)
- Tokens = matrix **entries** `(i,j)`, each carrying `value` + separable
  row/col positional encodings. (Row tokens were a half-measure: selection of
  the within-row column was smuggled through the absolute row position.)
- Axial / factorized attention (attend along rows, along columns) so cost is
  `O(n^3)` not `O(n^4)`, and the structure is explicit.
- **Equivariances for free**: permutation-equivariant over rows and over columns
  separately; variable token count -> **size-agnostic (cross-N)**. This is the
  property that lets a model trained on N<=4 run on N=6,7 — the real prize.

## Operator head
- Per step, predict an **operator** from a chosen family + which tokens it acts
  on (via attention, not external indexing):
  - elementary (rank-1) update — the verifiable first family;
  - Givens/Householder rotation — orthogonal canonicalization;
  - or a general (low-rank / full) linear map — the fully general case.
- Compose: `A <- M_t A` (or `M_t A M_t^{-1}` for similarity tasks). The composed
  `M_T...M_1` *is* the predicted transformation matrix — read it off directly.

## Supervision: end-state, not readout
- Loss = distance of the rolled-out state to the **target form**, e.g.
  `||A_T - target||` — no determinant readout, no pivoting.
- Early training may teacher-force against a reference decomposition (we can
  generate one), annealed to free rollout.

## Task instances (verifiable -> general)
1. **Reduce-to-identity (Gauss-Jordan):** `A -> I`; composed transform = `A^{-1}`.
   End-state `||A_T - I||`. General (inversion), verifiable, no det scaffolding.
2. **Canonicalize:** drive to diagonal / triangular via end-state `||off-form||`;
   det, inverse, eigenvalues all fall out as *readouts*, none hard-coded.
3. **Operator regression:** given `(A, B = T A)`, predict `T` by the rollout —
   the most general "predict the full transformation matrix".

## Metrics
- End-state error `||A_T - target||`; composed-operator error vs reference.
- **Test-time-compute scaling**: end-state error vs number of rollout steps
  (now adaptive — stop when the objective is met).
- **Cross-N generalization**: train N<=4, evaluate N=6,7 (impossible for the
  fixed-size MLP; the entry-token transformer's native test).

## Risks
- Credit assignment without per-step targets — mitigate with a reference
  decomposition for early teacher forcing.
- General operators may not be contractive — keep the elementary family as the
  first rung, generalize once the substrate trains.
- Per-step re-encoding cost — amortize (encode once per stage) for the big N.
