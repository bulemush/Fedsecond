# FedProxy-MA 服务器训练说明

## 实验类型与适用范围

本组配置是**任务级数据异质性（non-IID）实验**。configs/ma/base.yaml 继承 configs/experiments/llama2_full_len320_mb4.yaml，后者经 configs/experiments/llama2_full_multigpu.yaml 继承 configs/llama2_heterogeneous_50.yaml 和 configs/base.yaml。最终 data.scenario 为 heterogeneous，共有 8 个客户端和 8 个任务：OBQA、ARC-Easy、ARC-Challenge、CommonsenseQA、SST-2、MRPC、RTE、MNLI。当前数据划分将一个完整任务分给一个客户端；它不是同一任务在各客户端独立同分布的同质性实验，也不代表所有形式的 non-IID 分布。

MA 路径沿用现有 fedproxy-repro 安装与依赖，不需要更换服务器中的 conda 环境，也未在 pyproject.toml 中增加依赖。MA 配置继承原 FedProxy 的 320-token 诊断配置，但在 configs/ma/base.yaml 中将每任务训练样本上限设为服务器现有基线的 3000，并将已删除的原模型目录声明为迁移到当前重新下载的模型目录。代理压缩、优化器、联邦轮数和评测设置继续继承。MA 另外改变 LoRA 层的训练与上传范围、上传坐标及服务器对部分观测的处理。

现有代码实现了部分 LoRA 层训练、共享轮转锚点、分层上传调度、A/B Top-k、可选公共探针、带来源轮次的客户端残差、FP32 稀疏包（int32、varint、Golomb/Rice 或 bitmap 索引）、掩码感知的相似度与符号聚合、客户端暂存状态、按轮恢复和成本报告。默认 MA base 现在使用 bitmap 上行、完整 P＋仅训练层 G 的序列化下行，以及精确零/常量状态省略；实际包长包含头部。自适应保留率使用上一轮各客户端训练任务损失的等权平均值，**不是公共开发集损失**；提供的主实验配置使用固定保留率。真实网络传输和全局 P 的增量缓存同步尚未实现。新传输的使用与兼容说明见 [lossless_communication.md](lossless_communication.md)。

调度器按每个客户端自己的上传窗口轮转。在默认 16 个代理层、8 个客户端的 Conservative 与 Balanced 预算下，每轮每个非锚点层至少有 2 个上传客户端，训练但未上传的层在残差 TTL=4 轮内重新获得上传机会。其他层数和客户端数量必须先通过调度检查。coverage_window > 1 尚未实现，配置校验会拒绝。dry-run 输出全程残差调度检查；正式运行将结果保存到 metrics/schedule_audit.json。right_censored_pending_layers 包含训练结束前来不及再次上传的层，不能把这部分训练计算与未发送更新忽略掉。获得层级上传机会也不保证每个低幅值坐标一定进入 Top-k。

## 在原 conda 环境中先做验证

在服务器的项目根目录执行；以下命令不创建或切换 conda 环境：

    PYTHONPATH=src pytest -q tests/test_reuse.py tests/test_ma_config.py tests/test_ma_protocol.py tests/test_ma_report.py tests/test_ma_tiny_integration.py
    PYTHONPATH=src python -m fedproxy.cli smoke --config configs/smoke.yaml

第一条检查基线复用、MA 配置、协议、报告及无需下载数据的两轮 tiny 模型训练；第二条回归检查原 FedProxy smoke。任一检查失败时，先排查，不要启动完整 7B 训练。

## 复用相同的基线代理与数据划分

configs/ma/base.yaml 默认复用 runs/llama2_full_len320_mb4_seed42 的已有压缩代理和数据划分。该基线 `resolved_config.train.yaml` 记录的旧模型目录已被删除，当前模型与分词器均从 `/home/async/data-disk/wjf/wgh/Fedsecond/data/models` 加载。`run.reuse_model_relocation` 明确记录旧目录和新目录，供复用检查及结果报告识别；检查会核对新目录的模型层数、主要结构字段和分词器词表哈希是否与基线压缩清单一致，不加载 7B 权重。由于旧权重已删除，无法逐字节证明新下载的权重与基线完全相同；跨这两次下载的严格权重同一性仍依赖原模型修订版本的外部记录。所有 MA 变体都继承每任务 3000 条样本的设置。若基线不存在，先建立与 MA 设置一致的基线；仓库现有 `configs/experiments/llama2_full_len320_mb4.yaml` 默认为 5000 条/任务，不能直接用它产生可比较的 3000 条基线。

    test -r /home/async/data-disk/wjf/wgh/Fedsecond/data/models/config.json && echo "当前模型目录可读取"

开始后台任务前，先在项目根目录运行一次无需加载 7B 模型的复用配置检查：

    PYTHONPATH=src python -m fedproxy.reuse \
      --config configs/ma/dense.yaml \
      --source runs/llama2_full_len320_mb4_seed42

若提示 `Reuse configuration compatible`，即可启动相应配置的 `reuse-all`。若列出其他配置差异，输出会同时显示 MA 值、基线值及读取的 `resolved_config.train.yaml`（无此文件时为 `resolved_config.compress.yaml`）。请按实际基线设置修正 MA 配置继承链，或将 `run.reuse_artifacts_from` 指向设置匹配的基线。模型迁移声明只针对这次已知的旧目录与新目录；换成其他模型目录或分词器会被拦截。服务器上可用 `cat runs/llama2_full_len320_mb4_seed42/resolved_config.train.yaml` 核对基线快照。已有指向正确基线的符号链接可直接复用，无需删除或重新压缩。

每个 MA 配置只需启动一条后台命令。reuse-all 会在**同一个 nohup 进程中**依次执行复用基线产物与截断审计、dry-run、train、fuse、evaluate-fused；某一步失败即停止后续步骤。以 Dense-MA 为例：

    bash scripts/launch_experiment.sh configs/ma/dense.yaml 0,1 reuse-all

其中 0,1 是可见的物理 GPU 编号，请按服务器实际情况替换。该命令会打印 PID 与日志路径；按 Ctrl+C 只停止当前终端的日志查看，后台流程继续。确认 Dense-MA 完成并查看融合评测结果后，再对下一配置运行，例如：

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

- metrics/aggregation.json 保存每轮客户端训练指标、数据包 ID、实际上行包字节、新协议实际下行包字节，以及完整 FP32 P/G 的参考估算字节、有效相似度对及坐标共同观测数量。metrics/round_XXXX.json 保存 checkpoint 提交后的整轮墙钟时间。若提交后、时间文件写入前退出，报告将墙钟时间视为缺失。
- MA 上行以序列化后的 update.bin 长度计数；新协议下行使用 communication_bytes.downlink（完整包长，含头部），旧实验无该字段时仍使用 downlink_fp32_state_estimate。新实验中后者仅为未优化完整 P/G 的参考值。原 FedProxy 基线按原始 FP32 张量负载计数，尚无真实网络协议计量。报告增加 downlink_accounting 区分口径；统计已提交轮次，未计失败轮次重试和网络封装开销。
- 默认 worker 与原实现一样，针对每个客户端重新加载代理模型。减少可训练 LoRA 层不保证端到端时间同比例下降；应在服务器实测设备训练时间、联邦轮次墙钟时间和峰值显存。
- Conservative 默认训练 16 层中的 12 层、上传 10 层，非锚点 A/B 坐标各保留 80%；Balanced 分别为训练 8 层、上传 5 层、A/B 保留 50%/60%。实际代理层数由 dry-run 检查，不要假定其他压缩率也得到 16 层。
- 320-token 配置是继承原实验的诊断设置，不是论文附录中的 64-token 设置。
