# FedProxy Reproduction

这是依据用户提供的 `FedProxy.pdf`（arXiv:2604.19015v1）和实施规格完成的方法级复现。它实现 BI block pruning、LoRA A/B 空间的 PCR 与 H-TIES、单机串行客户端隔离、轮次恢复，以及合并 LoRA 后的训练自由层回填。

本项目没有作者源码背书。论文未披露或自相矛盾的细节均记录在 `docs/reproduction_decisions.md`；实现完成不代表论文表格中的数值已经复现。

## 安装与离线验证

在服务器上先安装与 CUDA/驱动相符的 PyTorch，再执行：

```bash
cd fedproxy-repro
python -m pip install -e '.[dev]'
pytest -q
python -m fedproxy.cli smoke --config configs/smoke.yaml
```

smoke 完全离线：随机初始化 4 层 tiny Llama、BI 删除一半层、2 客户端 × 2 轮、PCR/H-TIES、LoRA merge、非连续层回填、保存重载，并明确标记 `synthetic`。

## 真实复现流程

```bash
python -m fedproxy.cli prepare-data --config configs/llama2_heterogeneous_50.yaml
python -m fedproxy.cli compress --config configs/llama2_heterogeneous_50.yaml
python -m fedproxy.cli train --config configs/llama2_heterogeneous_50.yaml --rounds 10
python -m fedproxy.cli train --config configs/llama2_heterogeneous_50.yaml --rounds 10 --resume runs/llama2_heterogeneous_50/checkpoints/round_0005
python -m fedproxy.cli fuse --config configs/llama2_heterogeneous_50.yaml --checkpoint runs/llama2_heterogeneous_50/checkpoints/round_0010
python -m fedproxy.cli evaluate --config configs/llama2_heterogeneous_50.yaml --model-kind fused
python -m fedproxy.cli compare --runs-root runs --output reports/comparison.csv
```

`--rounds 10` 只是运行示例，不是论文披露值。真实训练会先写出 `training_budget.json`。所有客户端在每轮都从相同 `round_base` 开始；每轮完成后原子保存 adapter、conflict 和 schema metadata。

## 主要实现边界

- 压缩只有论文明确给出的 BI 结构化剪枝，没有臆造 KL 蒸馏。
- 聚合在 LoRA A/B 张量空间工作；effective-weight 聚合不是等价替换，未混入主方法。
- 默认 global adapter top-k；tensor top-k 只作为显式变体。
- 首轮冲突为零；本轮未裁剪 update 计算出的冲突只用于下一轮 PCR。
- 原 LLM 的 embedding、final norm、lm_head 不参与训练且在融合时保留。
- 不包含 OT/FedOT/FedBiOT 的伪实现，也不宣称提供差分隐私或密码学 IP 保护。

详细环境见 `docs/server_setup.md`，公式索引见 `docs/paper_mapping.md`，已运行/未运行状态见 `docs/experiment_status.md`。

