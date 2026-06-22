# Future directions — operator-algebra transformer

Parking lot for research directions on the operator-algebra / determinant work.
See FINDINGS.md for results so far.

## 1. Depth scaling of the transformer
We've run the operator-algebra transformer at a fixed `depth=4`. Open question:
how does **transformer depth** trade against the two residuals we found —
per-step operator precision and autoregressive rollout drift (worse at larger N,
e.g. N=8)? Note this is *parallel* depth (more encoder layers), unlike the old
unrolled elimination model where effective depth came from BPTT through the
rollout and was the wall. A clean depth × N sweep would show whether depth buys
precision / longer stable rollouts / larger N, and where it saturates.

## 2. Second-order optimizer for plasticity
The operator transformer was finicky to train (needed lr=1e-4 + a final
LayerNorm just to optimize; the "emergent thinking" end-state objective got
stuck in a det-non-preserving local minimum). Both smell like ill-conditioning.
Try a **second-order / preconditioned optimizer** (K-FAC, Shampoo, Sophia) for
(a) trainability/robustness without the hand-tuned recipe, (b) escaping the
emergent-thinking local min, and (c) **plasticity** — keeping the model able to
acquire new sentences without the optimization collapsing. Plasticity is the
bridge to direction 3.

## 3. New-task acquisition: LoRA vs full finetune vs in-context
With the shared token space, a *new* algorithm/sentence can be acquired three
ways — compare them head to head on a held-out sentence:
- **In-context learning** — no weight change; the [ICL] route. Cheapest,
  but limited to what's inferable from context (and to function-class tasks).
- **LoRA finetuning** — small low-rank weight update toward the new task.
  Cheap, and test for interference / catastrophic forgetting of DET/INV/ICL.
- **Full finetuning** — upper bound on adaptation, baseline for forgetting.
Axes: data efficiency, compute, final accuracy, and forgetting of the original
sentences. This directly extends the multitask result (one model, DET+INV) and
the ICL result — it asks *how best to grow the operator algebra*.
