# MA 单任务 IID 实验

本套件使用现有 FedProxy conda 环境，不新增依赖。八个任务各自训练一个融合模型；每项最多取 3000 条训练样本，然后 IID 分到 4 个客户端。输入/目标长度为 320/128 token，本地 epoch 固定为 5，聚合 10 轮，micro batch 4、梯度累积 4，每客户端有效 batch 为 16，训练与划分种子均为 42。数据不足 3000 条时使用全部训练样本。320-token、5 epoch 设置标为 diagnostic。

公共模板为 `configs/ma/homogeneous/base.yaml`，八项任务配置位于同一目录。整个 `ma` 节继承当前 `configs/ma/base.yaml`，包括训练/上传/锚点层数 12/10/1、80% A/B 保留率、bitmap 索引、active_conflict 下行、零/常量状态省略、时间戳残差和 MA 聚合。模板自身不单独启动。

## 启动前检查

在服务器项目根目录、已激活的原 conda 环境中执行：

```bash
PYTHONPATH=src python -m fedproxy.ma.homogeneous validate
```

检查成功会为八个任务分别输出 `[ma-homogeneous] validated task=... epochs=5 clients=4 ...`，进程退出码为 0。该阶段不加载模型权重或训练，不创建任务输出；发生已声明的旧模型目录迁移时会加载分词器核对词表哈希。

每个任务的基线默认来自 `runs/llama2_homogeneous_<task>_len320_mb4_seed42`，必须已有完整的 10 轮 checkpoint、代理、数据划分、最终融合模型及其单任务评测。校验读取基线的实际 `resolved_config.train.yaml`，比较模型、压缩、数据、LoRA、优化器、epoch、batch、轮次、PCR、聚合配置、评测设置和种子。预算不同会逐字段报错，不能用修改历史快照的方式跳过。

公共配置清空了异质性实验的复用来源和模型迁移声明。若基线记录的模型路径正好符合 `configs/ma/base.yaml` 已声明的旧→新迁移，套件将验证结构与分词器词表，并将迁移记录写入该任务的冻结配置。未知模型路径差异仍会报错；旧权重已删除时，结构和词表检查不等于权重逐字节相同的证明。

各任务复用自己基线的 `compression/` 和 `data_manifest.json`，从相同初始 LoRA 开始新训练；不从基线最终 LoRA 或旧客户端残差继续训练。基线的任务内样本索引和客户端划分原样复用。

## 全流程后台运行

```bash
bash scripts/launch_ma_homogeneous_suite.sh 0,1 all
```

命令以一个 nohup 后台进程依次运行 OBQA、ARC-Easy、ARC-Challenge、CommonsenseQA、SST-2、MRPC、RTE、MNLI。每个任务内部使用两张 GPU，并行训练两个客户端，分两批完成四客户端一轮；八个任务按顺序执行。日志查看界面按 Ctrl+C 后，后台进程继续。

默认输出结构：

```text
runs/MA_homogeneous/
  logs/
  llama2_homogeneous_obqa_len320_mb4_seed42/
  llama2_homogeneous_arc_easy_len320_mb4_seed42/
  llama2_homogeneous_arc_challenge_len320_mb4_seed42/
  llama2_homogeneous_commonsense_qa_len320_mb4_seed42/
  llama2_homogeneous_sst2_len320_mb4_seed42/
  llama2_homogeneous_mrpc_len320_mb4_seed42/
  llama2_homogeneous_rte_len320_mb4_seed42/
  llama2_homogeneous_mnli_len320_mb4_seed42/
  summary.json
  comparison.csv
```

全部所选任务在训练前完成基线检查，并保存各自 `execution_config.yaml`。每项依次执行复用/截断审计、dry-run、训练、融合、单任务 evaluate-fused、严格配对比较。正式 dry-run 检查实际代理层数、四客户端调度与残差上传机会。使用默认 16 层代理时每轮非锚点层有 2～3 个上传客户端。

已有正确的压缩和数据符号链接可复用；它们引用的原基线目录在实验期间应保持可读。套件不会移动截图中的原 FedProxy 目录。

## 单任务与中断恢复

只运行 ARC-Challenge：

```bash
bash scripts/launch_ma_homogeneous_suite.sh 0,1 arc_challenge
```

中断后重新执行原启动命令。已完成任务会验证最终 checkpoint、融合清单与评测绑定后跳过；部分任务从最后一个完整 checkpoint 继续。如果尚未提交第一轮，则重新开始该轮，既有合法客户端包由原 MA 事务逻辑重放。轮数完成但融合或评测未完成时，套件补做相应阶段。配置或残差状态不匹配会停止，不会把其他预算的产物标为当前结果。

套件使用进程锁，防止同一 `MA_homogeneous` 目录同时运行两次套件；进程退出或被终止后系统会释放锁。日志路径与 PID 在启动时打印。使用这个入口运行单项和全套，避免同时手工启动同一任务。

恢复需要保留任务目录中的 `clients/` 残差状态、checkpoint、符号链接及 `execution_config.yaml`。已启动的任务冻结配置不能修改；更改配置后需要新的输出目录。仅停止日志查看不等于终止训练。

## 查看结果

单任务评测：

```bash
cat runs/MA_homogeneous/llama2_homogeneous_arc_challenge_len320_mb4_seed42/evaluation/fused_summary.json
cat runs/MA_homogeneous/llama2_homogeneous_arc_challenge_len320_mb4_seed42/comparison.json
```

八项全部完成后自动生成汇总，也可重新读取已完成结果：

```bash
python scripts/summarize_ma_homogeneous_suite.py
cat runs/MA_homogeneous/summary.json
cat runs/MA_homogeneous/comparison.csv
```

`summary.json` 含 MA 各任务 accuracy、QA/GLUE/ALL 平均、本地 FedProxy 对应分数、差值（百分点）及逐项通信统计。`comparison.csv` 提供每任务分数和上行、下行、双向载荷比。QA 为四个 QA 模型各自任务分数的平均，GLUE 同理，ALL 为八项等权平均；不是同一个模型在八项任务上的结果。

比较要求每对任务的训练预算、数据划分、压缩清单和原始 lm-eval 任务配置/版本一致。缺少任意任务或评测过期时拒绝生成不完整平均。若基线评测缺少旧版本融合绑定，结果中的 `evaluation_checkpoint_verified` 会标为 false。若评测协议不一致，应在相同环境重新评测对应融合模型。

MA 通信以实际编码包长度计数，原基线以原始 FP32 状态计数，详细口径保留在任务报告中；成本统计排除初始化模型分发和真实网络封装。单种子配对结果只作描述性效果检查，不能证明统计非劣或实际计算时间下降。

## 验证代码

```bash
PYTHONPATH=src pytest -q tests/test_ma_homogeneous.py tests/test_ma_config.py tests/test_reuse.py tests/test_ma_report.py tests/test_homogeneous_report.py
```

服务器原环境中可补充运行现有 tiny 集成和协议测试：

```bash
PYTHONPATH=src pytest -q tests/test_ma_protocol.py tests/test_ma_tiny_integration.py
```

本地若未安装 PyTorch/Transformers，前一组仍可验证套件配置、预算拦截、续跑调度、结果身份与汇总；tiny 训练与真实 7B 实验需要在服务器环境验证。
