# FedProxy-MA 服务器训练说明

## 实验类型与适用范围

本组配置是**任务级数据异质性（non-IID）实验**。configs/ma/base.yaml 继承 configs/experiments/llama2_full_len320_mb4.yaml，后者经 configs/experiments/llama2_full_multigpu.yaml 继承 configs/llama2_heterogeneous_50.yaml 和 configs/base.yaml。最终 data.scenario 为 heterogeneous，共有 8 个客户端和 8 个任务：OBQA、ARC-Easy、ARC-Challenge、CommonsenseQA、SST-2、MRPC、RTE、MNLI。当前数据划分将一个完整任务分给一个客户端；它不是同一任务在各客户端独立同分布的同质性实验，也不代表所有形式的 non-IID 分布。

MA 路径沿用现有 fedproxy-repro 安装与依赖，不需要更换服务器中的 conda 环境，也未在 pyproject.toml 中增加依赖。模型、数据、代理压缩、优化器、联邦轮数和评测配置均继承原 FedProxy 的 320-token 全规模诊断配置。MA 配置改变 LoRA 层的训练与上传范围、上传坐标及服务器对部分观测的处理。

现有代码实现了部分 LoRA 层训练、共享轮转锚点、分层上传调度、A/B Top-k、可选公共探针、带来源轮次的客户端残差、FP32 稀疏包（int32、varint 或 Golomb/Rice 索引）、掩码感知的相似度与符号聚合、客户端暂存状态、按轮恢复和成本报告。自适应保留率使用上一轮各客户端训练任务损失的等权平均值，**不是公共开发集损失**；提供的主实验配置使用固定保留率。真实网络传输和稀疏下行同步尚未实现。

调度器按每个客户端自己的上传窗口轮转。在默认 16 个代理层、8 个客户端的 Conservative 与 Balanced 预算下，每轮每个非锚点层至少有 2 个上传客户端，训练但未上传的层在残差 TTL=4 轮内重新获得上传机会。其他层数和客户端数量必须先通过调度检查。coverage_window > 1 尚未实现，配置校验会拒绝。dry-run 输出全程残差调度检查；正式运行将结果保存到 metrics/schedule_audit.json。right_censored_pending_layers 包含训练结束前来不及再次上传的层，不能把这部分训练计算与未发送更新忽略掉。获得层级上传机会也不保证每个低幅值坐标一定进入 Top-k。

## 在原 conda 环境中先做验证

在服务器的项目根目录执行；以下命令不创建或切换 conda 环境：

    PYTHONPATH=src pytest -q tests/test_reuse.py tests/test_ma_config.py tests/test_ma_protocol.py tests/test_ma_report.py tests/test_ma_tiny_integration.py
    PYTHONPATH=src python -m fedproxy.cli smoke --config configs/smoke.yaml

第一条检查基线复用、MA 配置、协议、报告及无需下载数据的两轮 tiny 模型训练；第二条回归检查原 FedProxy smoke。任一检查失败时，先排查，不要启动完整 7B 训练。

## 复用相同的基线代理与数据划分

c
    bash scripts/launch_llama2_full_len320_mb4.sh 0,1

其中 0,1 是可见的物理 GPU 编号，请按服务器实际情况替换。启动器以 nohup 运行并显示日志；停止查看日志不等于停止后台训练。等基线完成后，再启动 MA 实验。

开始后台任务前，先在项目根目录运行一次无需加载 7B 模型的复用配置检查：

    PYTHONPATH=src python -m fedproxy.reuse \
      --config configs/ma/dense.yaml \
      --source runs/llama2_full_len320_mb4_seed42

若提示 `Reuse configuration compatible`，即可启动相应配置的 `reuse-all`。若列出 `model.name_or_path`、`data.max_input_length` 等差异，输出会同时显示 MA 值、基线值及读取的 `resolved_config.train.yaml`（无此文件时为 `resolved_config.compress.yaml`）。请按实际基线设置修正 MA 配置继承链，或将 `run.reuse_artifacts_from` 指向设置匹配的基线；不要仅为跳过校验而把不同模型或不同分词器视为同一模型。服务器上可用 `cat runs/llama2_full_len320_mb4_seed42/resolved_config.train.yaml` 核对基线快照。仅文件访问模式、压缩时的设备放置、是否执行截断审计，以及确实解析到同一文件的路径写法，可以不同；实验报告也按这一口径比较共享配置。已有指向正确基线的符号链接可直接复用，无需删除或重新压缩。

每个 MA 配置只需启动一条后台命令。reuse-all 会在**同一个 nohup 进程中**依次执行复用基线产物与截断审计、dry-run、train、fuse、evaluate-fused；某一步失败即停止后续步骤。以 Dense-MA 为例：

    bash scripts/launch_experiment.sh configs/ma/dense.yaml 0,1 reuse-all

该命令会打印 PID 与日志路径；按 Ctrl+C 只停止当前终端的日志查看，后台流程继续。确认 Dense-MA 完成并查看融合评测结果后，再对下一配置运行，例如：

    bash scripts/launch_experiment.sh configs/ma/partial.yaml 0,1 reuse-all

其余主实验配置分别使用：

    bash scripts/launch_experiment.sh configs/ma/rotate.yaml 0,1 reuse-all
    bash scripts/launch_experiment.sh configs/ma/sparse.yaml 0,1 reuse-all
    bash scripts/launch_experiment.sh configs/ma/base.yaml 0,1 reuse-all
    bash scripts/launch_experiment.sh configs/ma/balanced.yaml 0,1 reuse-all

以上命令应**逐项执行并检查结果，不要在同一组 GPU 上同时启动多个配置**。reuse-all 会为每个配置使用独立运行目录；不要对已有完整 checkpoint 的目录再次运行全流程。若基线目录路径不同，只修改 MA 配置的 run.reuse_artifacts_from。MA 报告会核对压缩清单和数据划分哈希。

建议按以下顺序逐项运行，每项使用自己的目录，检查前一项结果后再启动下一项：

- dense.yaml：全层训练和上传，无残差；隔离 MA 聚合定义本身的效果。
- partial.yaml：只减少训练层；检验部分训练的效果与实际计算时间。
- rotate.yaml：在部分训练基础上轮转上传层；检验层级通信压缩的影响。
- sparse.yaml：加入 A/B Top-k；检验坐标稀疏上传的影响。
- base.yaml：Conservative 配置，加入时间戳残差；检验延迟增量的恢复效果。
- balanced.yaml：进一步减少训练层、上传层和保留坐标；作为较强的降本候选。

可选消融包括 probe.yaml（从固定 Top-k 名额中保留 5% 公共探针）、golomb.yaml（保持消息数值相同，只改 Golomb/Rice 索引编码）和 adaptive.yaml（基于上一轮训练损失调整保留率）。不要在 Dense-MA 已明显退化时继续叠加压缩。320-token 运行标记为 diagnostic，不能直接与论文的 64-token 数字比较。

如果没有可复用的基线产物，MA 配置的 all 阶段可以独立准备数据并压缩：

    bash scripts/run_experiment.sh configs/ma/base.yaml 0,1 all

但仍须另跑同设置的原 FedProxy 基线。独立重新压缩所得代理未必完全相同；清单不匹配时，比较报告会拒绝合并结果。

## 后台训练与中断恢复

上述 reuse-all 命令已经包含完整流程的 nohup。如果此前已单独完成前置步骤，只想在后台继续 train 阶段，可运行：

    bash scripts/launch_experiment.sh configs/ma/base.yaml 0,1 train

完整轮次已保存 checkpoint 后中断，可恢复：

    bash scripts/run_experiment.sh configs/ma/base.yaml 0,1 resume

若第一轮 checkpoint 尚未生成，重新执行 train；已暂存的客户端数据包会按原字节重放。恢复所需的残差快照保存在 runs/<实验名>/clients/，请勿在 checkpoint 仍需使用时删除。已提交的轮次 checkpoint 是 ACK 边界。当前实现假定客户端状态在可信的共享工作区中，并采用同步全参与训练。

## 结果比较

候选配置完成 train、fuse 和 evaluate-fused 后，与原 FedProxy 基线比较：

    PYTHONPATH=src python -m fedproxy.cli ma-report \
      --baseline runs/llama2_full_len320_mb4_seed42 \
      --candidate runs/llama2_ma_conservative_seed42 \
      --output reports/ma_conservative_vs_fedproxy.json

报告要求训练和评测配置、数据划分、代理压缩清单、评测任务细节及轮数一致；它还检查融合模型是否来自纳入成本统计的最终 checkpoint，以及 MA 评测是否在该次融合后进行。检查失败时重新融合和评测。缺少融合哈希的旧基线评测仍可读取，但会标为未验证。

单个种子的报告只能作描述性比较，不能证明“保证 FedProxy 效果”。正式非劣结论至少需要 3 组匹配的基线与候选种子；每组种子相同、组间种子不同。完成后，可用 python -m fedproxy.cli ma-paired-report，按种子顺序分别重复传入 --baseline 和 --candidate，再指定 --output reports/ma_paired.json。该报告给出融合模型 ALL 指标的配对单侧 95% 下界（默认 1 个百分点非劣界值）及任务级检查；验收条件见 ../../FedProxy_MA_Experiment_Framework_Plan.md。参数数量下降不能替代计算时间实测。

## 资源与计量注意事项

- metrics/aggregation.json 保存每轮客户端训练指标、数据包 ID、实际上行包字节、完整 FP32 P/G 下行估算字节、有效相似度对及坐标共同观测数量。metrics/round_XXXX.json 保存 checkpoint 提交后的整轮墙钟时间。若提交后、时间文件写入前退出，报告将墙钟时间视为缺失。
- MA 上行以序列化后的 update.bin 长度计数。原 FedProxy 基线目前按原始 FP32 张量负载计数；下行是完整 P/G 的 FP32 状态估算，尚无真实网络协议计量。论文或图表中必须保留这些口径标签。
- 默认 worker 与原实现一样，针对每个客户端重新加载代理模型。减少可训练 LoRA 层不保证端到端时间同比例下降；应在服务器实测设备训练时间、联邦轮次墙钟时间和峰值显存。
- Conservative 默认训练 16 层中的 12 层、上传 10 层，非锚点 A/B 坐标各保留 80%；Balanced 分别为训练 8 层、上传 5 层、A/B 保留 50%/60%。实际代理层数由 dry-run 检查，不要假定其他压缩率也得到 16 层。
- 320-token 配置是继承原实验的诊断设置，不是论文附录中的 64-token 设置。
