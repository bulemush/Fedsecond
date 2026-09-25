# Experiment Result-to-Claim

## 2026-09-21 — LLaMA2-7B medium diagnostic

- Intended claim: reproduce the FedProxy method and ultimately test whether its reported heterogeneous 50% proxy performance can be reproduced.
- Result source: medium run with 512 samples/task, 128 calibration samples, 2 local epochs and 3 communication rounds.
- Observed: QA 0.4241351765, GLUE 0.5634710526, all-task average 0.4938031145.
- Paper reference: QA 0.6011 and GLUE 0.7944.
- Verdict: **partial** (provisional; no paper claim audit is available).
- Confidence: high.
- Supported: the independent method implementation executes end to end on LLaMA2-7B under a reduced heterogeneous setting and produces a usable diagnostic baseline.
- Not supported: numerical reproduction of the paper, equivalence of the undisclosed communication schedule, or a stable fused-model advantage.
- Missing evidence: larger/full runs, original/proxy/fused comparisons, fixed evaluation provenance, convergence across rounds, and at least three random seeds.
- Routing action: supplement. Run the two-GPU large diagnostic, then the full configuration and multi-seed/round sweeps if the large run is stable.

## 2026-09-25 — LLaMA2-7B large run, rounds 1–5 metadata

- Intended claim: numerically reproduce the paper's heterogeneous 50% proxy result.
- Result source: user-provided round 1–5 checkpoint metadata, compression manifest, and final original/fused evaluation.
- Observed: fused QA 56.69%, GLUE 69.86%, ALL 63.28%; paper QA 60.11%, GLUE 79.44%, ALL 69.77%. Original-model QA 46.11% and GLUE 56.20% are close to the paper's ZeroShot 46.00% and 54.94%.
- Verdict: **partial** (provisional; no paper claim audit is available).
- Confidence: high for the numerical gap and cross-round trends; medium for causal interpretation.
- Supported: the implementation improves on its original-model baseline, but remains 6.49 percentage points below the paper's ALL result. Across rounds, client weights converge toward 1/8 and next-conflict mean rises from 0.6754 to approximately 0.7255.
- Not supported: a claim that conflict, compression, or insufficient rounds alone caused the gap; training loss is not validation accuracy, and per-round global evaluation is absent.
- Missing evidence: per-round global checkpoint scores, training-versus-evaluation prompt/label alignment and truncation audit, equal-budget comparison with the paper, and multiple seeds.
- Routing action: supplement. First evaluate all five checkpoints and audit prompt/target alignment, then test paper-scale data and local-epoch budgets before changing the method.

## 2026-09-25 — Five-round fused global-checkpoint evaluation

- Intended claim: reproduce the paper's heterogeneous LLaMA2-7B result and determine whether additional communication rounds are the next priority.
- Result source: user-provided sequential zero-shot lm-eval output for round 1–5 fused global checkpoints; evaluation protocol fingerprint is identical across rounds.
- Observed: ALL 56.62% → 60.82% → 62.22% → 62.83% → 63.28%; final remains 6.49 percentage points below the paper. MNLI 50.31% → 49.06% → 49.73% → 47.76% → 46.39%. RTE peaks at 71.12% in round 4 and ends at 70.40%. OpenBookQA and ARC Easy are nearly flat.
- Verdict: **partial** (provisional; no paper claim audit is available).
- Confidence: high for the observed trajectories; medium for causal interpretation.
- Supported: aggregate performance improves across the five rounds; MNLI's trajectory diverges from overall improvement, and QA improvements are concentrated in CommonsenseQA.
- Not supported: full numerical reproduction, a causal claim that aggregation alone harms MNLI, or a claim that more rounds will reliably close the gap.
- Missing evidence: MNLI pre-round global versus client-only versus post-aggregate scores; rendered training and lm-eval prompt/target alignment; 64-token prompt truncation rate; task-level uncertainty and multiple seeds.
- Routing action: supplement. Audit MNLI and QA protocols and perform targeted pre/post aggregation comparisons before extending rounds or increasing training budget.
