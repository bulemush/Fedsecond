# Reviewer prompt

Evaluate the completed medium-scale FedProxy reproduction result against intended reproduction claims.

- LLaMA2-7B; 8 heterogeneous task-clients; 50% proxy.
- 512 training samples/task; 128 Alpaca calibration samples; 2 local epochs; 3 communication rounds.
- Result: QA 0.4241351765, GLUE 0.5634710526, all-task average 0.4938031145.
- Paper heterogeneous 50% reference: QA 0.6011, GLUE 0.7944.
- The paper does not disclose communication rounds.

Judge claim support, supported and unsupported conclusions, missing evidence, revised claim, next experiments, and confidence without inflating the evidence.
