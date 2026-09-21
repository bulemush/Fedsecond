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

数据默认采用本地优先策略。程序依次识别：

1. 标准持久化目录 `data/datasets/<任务>/<revision>/<split>`；
2. 手工放置的 Hugging Face `save_to_disk` 数据：`data/<任务>/`、`data/<任务>/<split>/` 或 `data/datasets/<任务>/`；
3. `data/<任务>/<split>.jsonl|json|csv|parquet`（训练集也支持 `data/<任务>.jsonl|json|csv|parquet`）。

本地均不存在时才从 Hugging Face 下载。原始下载缓存保存在 `data/huggingface`，解析后的 split 会统一持久化到标准目录，后续命令可以离线复用。可用 `--set data.local_dir=/data/fedproxy` 将数据根目录放到服务器数据盘。

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

## 一条命令运行实验

预置实验配置已经包含输出目录、样本数、local epochs、batch、轮数和单/多卡调度，不需要每次输入 `--set`：

```bash
chmod +x scripts/run_experiment.sh
chmod +x scripts/launch_experiment.sh

# 中等规模，物理 GPU 3
./scripts/run_experiment.sh configs/experiments/llama2_medium.yaml 3

# 较大规模，物理 GPU 0、1；客户端每批并行两个
./scripts/run_experiment.sh configs/experiments/llama2_large_multigpu.yaml 0,1

# 正式规模，物理 GPU 0、1、2、3；客户端每批并行四个
./scripts/run_experiment.sh configs/experiments/llama2_full_multigpu.yaml 0,1,2,3
```

通过 Xshell 启动长时间任务时，可使用后台入口；SSH 断开后任务继续运行，脚本会打印日志路径和 PID：

```bash
./scripts/launch_experiment.sh configs/experiments/llama2_large_multigpu.yaml 2,3 train
```

第三个参数可以只运行一个阶段：`prepare`、`compress`、`dry-run`、`train`、`resume`、`fuse`、`evaluate`。其中 `evaluate` 会依次评测 original、proxy 和 fused；也可用 `evaluate-original`、`evaluate-proxy` 或 `evaluate-fused` 单独评测。例如：

```bash
./scripts/run_experiment.sh configs/experiments/llama2_large_multigpu.yaml 0,1 dry-run
./scripts/run_experiment.sh configs/experiments/llama2_large_multigpu.yaml 0,1 resume
```

多卡训练采用客户端并行：每张可见 GPU 同时训练一个独立客户端，每个客户端仍从同一个 `round_base` 开始；一批客户端结束后由 CPU 服务端统一分析和聚合。`max_parallel_clients: 0` 表示自动使用全部可见 GPU。多卡配置在压缩阶段使用 `device_map: balanced`，将完整基础模型均衡分片到可见 GPU。

GPU 编号不写入 YAML：`CUDA_VISIBLE_DEVICES` 必须在 Python 导入 PyTorch 前设置，所以由启动脚本的第二个参数注入。除这个编号外，实验超参数全部保存在 YAML 中；脚本同时把物理编号记录进 `training_budget.json`。

## 主要实现边界

- 压缩只有论文明确给出的 BI 结构化剪枝，没有臆造 KL 蒸馏。
- 聚合在 LoRA A/B 张量空间工作；effective-weight 聚合不是等价替换，未混入主方法。
- 默认 global adapter top-k；tensor top-k 只作为显式变体。
- 首轮冲突为零；本轮未裁剪 update 计算出的冲突只用于下一轮 PCR。
- 原 LLM 的 embedding、final norm、lm_head 不参与训练且在融合时保留。
- 不包含 OT/FedOT/FedBiOT 的伪实现，也不宣称提供差分隐私或密码学 IP 保护。

详细环境见 `docs/server_setup.md`，公式索引见 `docs/paper_mapping.md`，已运行/未运行状态见 `docs/experiment_status.md`。
