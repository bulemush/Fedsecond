# FedProxy Reproduction

FedProxy-MA 的独立实验路径见 [服务器训练说明](docs/ma_training.md)。它沿用当前 FedProxy 的安装环境、模型和训练配置，并增加部分层 LoRA、客户端稀疏传输及掩码聚合。
每个 MA 配置可用 `bash scripts/launch_experiment.sh configs/ma/dense.yaml 0,1 reuse-all` 独立启动完整的 nohup 流程；将配置名换为 partial、rotate、sparse、base 或 balanced，并在前一个配置完成后再启动下一个。

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

后台入口会立即跟随日志并显示每轮、每批客户端、GPU、epoch、优化步、完成百分比、loss、耗时和预计剩余时间。默认约每 30 秒更新一次；按 `Ctrl+C` 只退出日志查看，已通过 `nohup`/`setsid` 启动的训练仍会继续。

第三个参数可以只运行一个阶段：`prepare`、`compress`、`dry-run`、`train`、`resume`、`fuse`、`evaluate`。其中 `evaluate` 默认只评测 fused；如需基线，仍可显式运行 `evaluate-original` 或 `evaluate-proxy`。例如：

```bash
./scripts/run_experiment.sh configs/experiments/llama2_large_multigpu.yaml 0,1 dry-run
./scripts/run_experiment.sh configs/experiments/llama2_large_multigpu.yaml 0,1 resume
```

多卡训练采用客户端并行：每张可见 GPU 同时训练一个独立客户端，每个客户端仍从同一个 `round_base` 开始；一批客户端结束后由 CPU 服务端统一分析和聚合。`max_parallel_clients: 0` 表示自动使用全部可见 GPU。多卡配置在压缩阶段使用 `device_map: balanced`，将完整基础模型均衡分片到可见 GPU。

GPU 编号不写入 YAML：`CUDA_VISIBLE_DEVICES` 必须在 Python 导入 PyTorch 前设置，所以由启动脚本的第二个参数注入。除这个编号外，实验超参数全部保存在 YAML 中；脚本同时把物理编号记录进 `training_budget.json`。

## 截断诊断：按顺序运行

保留原 `llama2_large_multigpu` 实验不变。两个消融配置沿用原实验的压缩代理与数据划分，分别写入独立目录；`reuse-all` 会检查并链接这两份产物，先审计，再训练、融合和评测。务必在服务器项目根目录运行，且先确认 `runs/llama2_large_multigpu/compression/proxy`、`compression_manifest.json` 与 `data_manifest.json` 都存在。

```bash
# 1. 先只审计 320-token 对照；确认各任务 truncation_rate 近似为 0
./scripts/run_experiment.sh configs/experiments/llama2_large_len320.yaml 2,3 reuse-audit
# 2. 审计通过后运行 320-token 对照
./scripts/run_experiment.sh configs/experiments/llama2_large_len320.yaml 2,3 reuse-all
# 3. 上一步完成后，先审计 64-token 结构化版本
./scripts/run_experiment.sh configs/experiments/llama2_large_structured64.yaml 2,3 reuse-audit
# 4. 审计通过后运行 64-token 消融
./scripts/run_experiment.sh configs/experiments/llama2_large_structured64.yaml 2,3 reuse-all
```

长时间任务可将上面两条命令中的 `run_experiment.sh` 改为 `launch_experiment.sh`（后台 `nohup`，可断开 Xshell）；仍应等第一项完成、检查结果后再启动第二项。审计写入 `runs/<实验名>/metrics/truncation_audit.json`，评测写入 `runs/<实验名>/evaluation/fused_summary.json`。64-token 结构化审计如发现答案提示、NLI 假设或选项标签丢失，会在训练前报错；不要绕过该检查。320-token 是诊断消融，不是论文 Appendix B 的 64-token 设定。比较原实验和两个新实验的各任务准确率时，还需注意评测采用 lm-eval 的独立提示模板，训练提示的截断率不能直接解释全部评测差距。

## 全规模 320-token、micro batch 4 实验

`configs/experiments/llama2_full_len320_mb4.yaml` 继承全规模配置：最多 5,000 条/任务、10 个本地 epoch、10 轮、多卡客户端并行；输入上限为 320 token，每卡 micro batch 为 4，累积 4 步，有效 batch 仍为 16。独立输出目录是 `runs/llama2_full_len320_mb4_seed42`。由于论文附录使用 64-token 输入，本实验仍标记为 diagnostic。在服务器项目根目录运行：

```bash
bash scripts/launch_llama2_full_len320_mb4.sh 0,1
```

脚本使用现有 `nohup` 入口，依次准备数据、截断审计、压缩、预算检查、训练、融合，并仅评测 fused。运行中可断开 SSH；终端自动显示日志，按 Ctrl+C 只停止查看。第二个参数可指定单个阶段，例如 `bash scripts/launch_llama2_full_len320_mb4.sh 0,1 resume`。完整流程拒绝覆盖已有数据、压缩产物或 checkpoint。原先 2,000 条样本的 320-token 零截断结论不适用于新增样本，须查看本实验的 `metrics/truncation_audit.json`。micro batch 4 在 320-token 下的峰值显存需由服务器实测，不能仅根据旧运行的显存占用推断。

## 数据同构实验：8 个任务各训练一个模型

`configs/experiments/llama2_homogeneous_*_len320_mb4.yaml` 提供八个单任务配置。每个任务最多取 3,000 条训练样本，按固定随机种子 IID 分到 4 个客户端；不足 3,000 条的任务使用全部可用训练样本。配置为 320-token、每轮本地 5 epoch、10 聚合轮、micro batch 4、累积 4 步。多卡按客户端并行，GPU 编号仅在启动命令中指定。320-token 与 5 epoch 都不是论文附录的原始设置，因此此套结果标记为 diagnostic。

在服务器项目根目录执行：

```bash
bash scripts/launch_homogeneous_suite.sh 0,1
```

这会在 `nohup` 后台依次运行八个任务（而不是同时争用两张 GPU）：每项准备本任务数据、审计、训练、融合、只评测 `evaluate-fused`。首项 OBQA 生成压缩代理，其余任务只复用这个相同基础模型和压缩参数对应的代理；各任务的数据划分、训练、checkpoint 和融合模型相互独立。完成的任务可跳过，部分完成的任务会中止以免覆盖；可指定单项，例如 `bash scripts/launch_homogeneous_suite.sh 0,1 rte`。如果中途训练失败，先用 `bash scripts/run_experiment.sh configs/experiments/llama2_homogeneous_rte_len320_mb4.yaml 0,1 resume` 恢复，再手动执行该任务剩余的 `fuse`、`evaluate-fused` 阶段，并重新运行套件。第一项压缩产物在后续任务完成前不要移动或删除。

每项结果在其 `runs/llama2_homogeneous_<任务>_len320_mb4_seed42/evaluation/fused_summary.json`；全部结束后汇总为 `runs/llama2_homogeneous_suite_len320_mb4_seed42/summary.json`，QA/GLUE 平均值来自八个分别训练的模型。终端按 Ctrl+C 只停止看日志；可通过启动命令打印的日志路径继续 `tail -f`。如需额外评估 original 或 proxy，单独用 `scripts/run_experiment.sh <对应配置> 0,1 evaluate-original` 或 `evaluate-proxy`。

## MA 数据同质性实验

`configs/ma/homogeneous/` 提供与上述八个同质性 FedProxy 基线配对的 MA 配置：每任务最多 3000 条样本、4 个 IID 客户端、320-token、5 本地 epoch、10 轮、micro batch 4 × 累积 4；MA 设置完整继承当前 `configs/ma/base.yaml`。所有 MA 结果集中于 `runs/MA_homogeneous/`，每个任务独立训练、融合和评测。

```bash
PYTHONPATH=src python -m fedproxy.ma.homogeneous validate
bash scripts/launch_ma_homogeneous_suite.sh 0,1 all
```

各任务复用原有 FedProxy 目录中的代理和数据划分，验证会核对其实际配置与预算。套件以 nohup 顺序完成 MA 的训练、融合和评估，每项评估后直接进入下一任务，支持同一命令续跑。最后生成仅含 MA 成绩的 `summary.json` 和 `results.csv`。详细目录、单任务命令和恢复规则见 [MA 同质性训练说明](docs/ma_homogeneous_training.md)。

## 主要实现边界

- 压缩只有论文明确给出的 BI 结构化剪枝，没有臆造 KL 蒸馏。
- 聚合在 LoRA A/B 张量空间工作；effective-weight 聚合不是等价替换，未混入主方法。
- 默认 global adapter top-k；tensor top-k 只作为显式变体。
- 首轮冲突为零；本轮未裁剪 update 计算出的冲突只用于下一轮 PCR。
- 原 LLM 的 embedding、final norm、lm_head 不参与训练且在融合时保留。
- 不包含 OT/FedOT/FedBiOT 的伪实现，也不宣称提供差分隐私或密码学 IP 保护。

详细环境见 `docs/server_setup.md`，公式索引见 `docs/paper_mapping.md`，已运行/未运行状态见 `docs/experiment_status.md`。
