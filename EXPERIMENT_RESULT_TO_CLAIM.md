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
