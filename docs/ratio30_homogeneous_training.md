# Ratio30 同质性实验：FedProxy 与 MA 方案 B

本套件使用服务器原有 conda 环境，不新增依赖。八个任务分别为 OBQA、ARC-Easy、ARC-Challenge、CommonsenseQA、SST-2、MRPC、RTE、MNLI；FedProxy 和 MA 各训练八个独立融合模型。

## 固定配置

| 参数 | 两个方法的共同设置 |
|---|---|
| 原始模型 | 当前服务器 `data/models` 中的 32 层 LLaMA-2-7B |
| 删除比例 | `compression.remove_ratio: 0.30` |
| 代理层数 | `32 - floor(32 × 0.30) = 23`，实际删除 9 层，即 28.125% |
| 训练数据 | 每任务最多 3000 条，IID 分给 4 个客户端；不足 3000 时使用全部 |
| 输入／目标长度 | 320／128 token |
| 本地 epoch／全局轮数 | 每轮本地 5 epoch，共 10 个聚合轮，全客户端参与 |
| Batch | micro batch 4 × 梯度累积 4，每客户端有效 batch 16 |
| 并行 | 两张 GPU，每卡一个客户端，四客户端分两批 |
| 优化器 | AdamW，学习率 5e-5，weight decay 0，恒定学习率，每客户端每轮重建优化器 |
| LoRA | rank 32、alpha 64、dropout 0.1，模块列表与原配置相同 |
| PCR | 开启，lambda 1e-5，sum reduction |
| 压缩校准 | Alpaca 512 条、长度 512、batch 1、seed 42 |
| 随机种子 | 划分、训练、全局 LoRA 初始化均为 42 |
| 评测 | 最终融合模型，只评对应任务，zero-shot、accuracy、batch 1 |

320-token 和 5 本地 epoch 沿用既定实验预算，仍标为 diagnostic。FedProxy 使用原 H-TIES 路径，训练 23 层代理的全部 LoRA 参数，骨干冻结。MA 使用方案 B：`train_blocks=17`、`upload_blocks=14`、`anchor_blocks=1`、`q_block=2`。这分别约为代理层数的 73.9% 和 60.9%，接近原 16 层代理的 12/16 和 10/16。

其余 MA 参数继承 `configs/ma/base.yaml`：A/B 保留率 0.8、固定稀疏率、探测率 0、bitmap 编码、active_conflict 下行与零／常量状态省略、残差 TTL 4 和衰减 0.1、masked_sum PCR、原 MA 聚合参数。每个非锚点层每轮至少两个客户端上传；坐标级稀疏率仍为 80%，该层覆盖条件不等于每个坐标都由两个客户端上传。

## 配置与目录

FedProxy 配置位于 `configs/experiments/homogeneous_ratio30/`，MA 配置位于 `configs/ma/homogeneous_ratio30/`。每组有一个公共模板和八项任务 YAML。通过套件入口启动任务，公共 `base.yaml` 不单独启动。

默认输出如下，原 ratio50 目录保持原样：

```text
runs/homogeneous_ratio30/
  shared/
    execution_config.yaml
    resolved_config.compress.yaml
    compression/
      compression_manifest.json
      proxy/
  FedProxy/
    llama2_homogeneous_<task>_len320_mb4_seed42/
      execution_config.yaml
      data_manifest.json
      compression -> ../../shared/compression
      checkpoints/round_0001/ ... round_0010/
      exports/
      evaluation/fused_summary.json
      result.json
    summary.json
    results.csv
  MA/
    llama2_homogeneous_<task>_len320_mb4_seed42/
      execution_config.yaml
      data_manifest.json -> 对应 FedProxy 任务的划分
      compression -> 同一个 shared/compression
      clients/
      checkpoints/
      exports/
      evaluation/fused_summary.json
      result.json
    summary.json
    results.csv
  experiment_manifest.json
  logs/
```

共享代理只生成一次，两个方法的所有任务引用同一份初始代理。每个任务的数据划分只准备一次，MA 原样引用对应 FedProxy 目录中的划分。全局 LoRA 初始化前显式重设 seed 42，避免准备步骤对随机状态的影响。初始化不加载任何旧任务最终 adapter，也不复用 ratio50 的 16 层代理或已训练 checkpoint。

`experiment_manifest.json` 记录共享压缩清单哈希、各任务划分哈希和冻结配置哈希，用于恢复时检查输入一致性。初始代理只占一份磁盘空间，各任务的最终融合模型独立保存。

## 启动

在服务器项目根目录、已经激活原 conda 环境后先检查：

```bash
PYTHONPATH=src python -m fedproxy.ratio30 validate --gpu-ids 2,3
```

首次检查验证两组配置、原始模型文件及层数，不生成代理、不训练、不创建实验输出。已有实验还会核对冻结配置、输入链接、划分、checkpoint 和残差；不读取另一个方法的成绩。成功应打印 `validated ... proxy_layers=23 MA=17/14/1 q_block=2`。

全流程后台运行两个方法：

```bash
bash scripts/launch_ratio30_homogeneous_suite.sh 2,3 both all
```

先生成并验证 23 层共享代理，再准备所选任务的划分；随后依次执行八个 FedProxy 任务和八个 MA 任务。每项内部依次做截断审计、dry-run、训练／续训、最终融合、evaluate-fused、保存自身结果，然后进入下一任务。过程没有 FedProxy 与 MA 的成绩比较拦截。

单独运行一个方法或一个任务：

```bash
bash scripts/launch_ratio30_homogeneous_suite.sh 2,3 fedproxy all
bash scripts/launch_ratio30_homogeneous_suite.sh 2,3 ma all
bash scripts/launch_ratio30_homogeneous_suite.sh 2,3 both arc_challenge
```

仅运行 MA 时，套件会在 FedProxy 子目录中准备必要的初始代理链接和数据划分，随后只训练 MA；不要求 FedProxy 已完成训练或评测。

启动器打印后台 PID 和日志路径，并打开日志查看。Ctrl+C 只停止查看，nohup 后台作业继续。GPU 编号可替换，但必须是两张不同且可见的卡。客户端训练阶段双卡各跑一个客户端；压缩、审计、融合和评估按原流程执行，不要求它们同时占满两张卡。23 层代理比原 16 层代理更大，dry-run 使用实际代理检查调度，真实显存与耗时以服务器运行记录为准。

第四个参数可指定新的总目录，两个方法及共享资源一起切换：

```bash
bash scripts/launch_ratio30_homogeneous_suite.sh 2,3 both all runs/homogeneous_ratio30_repeat
```

## 中断恢复

重复原启动命令即可。已完成且评测绑定正确的任务跳过；部分任务从最后一个完整 checkpoint 继续。第一轮尚未提交时重跑第一轮，MA 已暂存的有效上传包按原事务机制重放。FedProxy 从完整轮次边界恢复；轮内未提交的本地训练会重新执行。融合或评测尚未完成时补做相应阶段。

同一总目录使用统一进程锁。不要同时通过不同入口操作同一任务。保留原始模型、共享代理、`experiment_manifest.json`、冻结配置、数据划分、checkpoint，以及 MA 的客户端残差状态。配置或输入变化会停止恢复；改变预算时使用新目录。

已完成的压缩不会重做。压缩中断且尚未产生最终清单时，只有冻结配置一致且尚无任务 checkpoint 才允许重新执行压缩。已有清单但权重不完整、数据划分 JSON 损坏或来源不明的产物会报错，应根据具体报错恢复对应文件。

## 结果

每个方法的八项任务全部完成后，分别生成其 `summary.json` 和 `results.csv`。后续可独立重新汇总：

```bash
python scripts/summarize_ratio30_homogeneous_suite.py
python scripts/summarize_ratio30_homogeneous_suite.py --method fedproxy
python scripts/summarize_ratio30_homogeneous_suite.py --method ma
cat runs/homogeneous_ratio30/FedProxy/results.csv
cat runs/homogeneous_ratio30/MA/results.csv
```

使用自定义总目录时，在汇总命令后添加 `--root <总目录>`。CSV 提供任务分数、百分制 accuracy、轮数、本地 epoch、通信字节和耗时记录；JSON 含八项分数及 QA／GLUE／ALL 平均。每个平均值来自八个分别训练的模型。两个方法独立汇总，不要求评测元数据里的模型目录字符串相同。

FedProxy 统计原始 FP32 状态通信量，MA 优先统计实际编码包长度；均不包含初始化模型分发和网络协议封装。层选择只减少部分可训练 LoRA／优化器状态，不能把层预算比例直接当作显存或总计算时间降低比例。

## 验证

无需 CUDA 的配置、调度与流程检查：

```bash
PYTHONPATH=src pytest -q tests/test_ratio30.py tests/test_reuse.py tests/test_reuse_script.py tests/test_ma_homogeneous.py tests/test_ma_round_extension.py
```

服务器原 PyTorch 环境中可进一步运行真实张量协议与 tiny 训练测试：

```bash
PYTHONPATH=src pytest -q tests/test_ma_protocol.py tests/test_ma_transport_resume.py tests/test_ma_tiny_integration.py
```

纯 Python 流程测试使用替代训练器，不代表已完成真实 7B 双卡训练。最终准确率、显存和训练耗时需要在服务器上测量。
