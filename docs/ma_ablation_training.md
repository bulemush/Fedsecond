# ratio30 异质 MA 四组单种子消融

复用已完成的 `runs/heterogeneous_ratio30/MA`，只训练 A1–A4。固定 seed 42、23 层代理、每客户端训练 17 层、上传 14 层、1 个共享锚点、非锚点 A/B 保留率 0.8。各组从初始代理和空残差开始训练 10 轮，每轮本地 5 epoch、每任务最多 3000 条。每组约 1.5 天的估计对应四组串行约 6 天，融合和评测耗时另计。

| 组别 | 唯一干预 |
| --- | --- |
| A1 | 余弦使用双方全部实传坐标的范数，交集有效性和可靠度保留。全局更新对实际观测的有效非零贡献按权重求平均，正负双方均计入分母。冲突图继续按筛选前的观测生成。 |
| A2 | `ma.residual.enabled=false`，只选当前轮的增量。 |
| A3 | `pcr.lambda_reg=0`，继续计算并下发冲突图 G。 |
| A4 | 非锚点逐张量随机无放回选 k，k 与完整 MA 相同。随机种子包含客户端 ID、轮次、张量名和用途标签。锚点仍稠密，保留残差。 |

每个配置从已完成 MA 的 `resolved_config.train.yaml` 深拷贝生成，运行时保存成各组自己的 `execution_config.yaml`。该快照记录具体干预及参考配置、数据、压缩清单、十轮调度和评测产物的指纹。

## 检查与启动

在服务器项目根目录、原训练环境内执行。GPU 编号按实际情况替换。先做只读核对，它会读取已有 MA 的十轮 checkpoint、数据清单、调度及绑定到第十轮融合模型的八任务评测。

```bash
PYTHONPATH=src python -m fedproxy.ma.ablations validate
```

如果提示旧 MNLI 模板或评测绑定过期，按错误信息对已导出的融合模型重新评测，再核对。已有 MA 的真实训练设置须与本方案匹配。检查不会重新训练参考 MA。

依次运行四组，每组自动完成训练、融合和八任务评测。后台入口会输出 PID 和日志路径。

```bash
bash scripts/launch_ma_ablation_suite.sh 0,1
```

也可只运行一组。

```bash
bash scripts/launch_ma_ablation_suite.sh 0,1 A3
```

参考 MA 和新输出目录可作为第 3、4 个参数指定。

```bash
bash scripts/launch_ma_ablation_suite.sh 0,1 all runs/heterogeneous_ratio30/MA runs/heterogeneous_ratio30/ablations
```

前台运行使用 `scripts/run_ma_ablation_suite.sh`，参数相同。四组按 A1、A2、A3、A4 顺序运行。任一步失败即停，修复后使用原命令恢复。流程复用已完成并核验过的组，未完成的组从最后完整 checkpoint 恢复，保留原 MA 的 pending 包重放和残差提交逻辑。输出根目录上的运行锁避免同一套实验被重复启动。

如果 A1 已完成评测，但旧版在汇总时因 `harness_detail_hash mismatch` 停止，更新代码后可先用下文的 `summarize` 核对。lm-eval 为每组记录的融合模型目录不同，汇总时会识别各组自己的 `metadata.pretrained`。A1 通过核对后，原启动命令会直接复用它的评测并进入下一组。如果 `invalid.A1` 仍列出任务配置差异，按列出的字段对齐到完整 MA 的评测设置。评测可以复用已有融合模型。

### 双卡滚动调度与 OOM 恢复

MA 每张卡最多同时训练一个客户端。空闲卡从本轮队列立即接下一位，服务器收齐全部有效上传包后按原客户端顺序聚合。GPU 派发顺序会随可用显存与完成时间变化。同一客户端仍从该轮已提交的全局 LoRA 和自己的上轮残差开始，使用原有的客户端种子。

默认要求实际空闲显存严格大于 14 GiB。调度器派发前检查空闲量，工作进程加载模型前再次检查。任一卡发生 CUDA OOM 时，程序退出该卡的工作进程以释放其 CUDA 上下文，失败客户端回到本轮队列。其他满足门槛的 GPU 继续处理队列。出错卡冷却 30 秒后重新检查显存，符合门槛后重新接单。没有可用卡时每隔约 10 秒复查，训练日志会报告等待状态。显存检查是瞬时值，外部进程可以之后再分配显存，因此 OOM 重试仍然有用。

如果实际训练峰值超过 14 GiB，可用运行时环境变量提高门槛。例如已测得客户端需要更多空闲显存时，可设置 `FEDPROXY_MA_MIN_FREE_GIB=16` 后沿用原命令启动。显存门槛与派发顺序都在配置哈希之外。更新到支持新调度的代码后，等旧进程退出，再用相同 GPU 参数与实验目录启动。已提交轮次从最后 checkpoint 继续，当轮已落盘的上传包和残差直接重放，未完成客户端重新训练。由于 GPU 型号、后端和实际硬件算子可能不同，跨卡重试后的数值一致性应以实际环境验收为准。

## 训练前后的核验

- 逐轮比对参考 MA 与新组的训练层、上传层和锚点名单。逐客户端比对已有记录中的优化步数、有效目标 token、可训练参数数和上传坐标数。
- 复用参考 MA 的初始代理和数据清单。训练启动时计算初始 adapter 哈希，若参考 MA 保留首轮 `update.bin`，读取其包头并核对 `base_hash`。没有保留包时，记录共享代理与初始化种子的证据等级。
- 各组独立保存 checkpoint 和本地残差。新增机制记录保存在输出目录，通信包保持原格式。
- 默认开关兼容已有配置。新增可选字段为 `ma.aggregation.similarity_scope`、`ma.aggregation.merge_rule` 和 `ma.compression.selection`，默认分别是 `intersection`、`dominant_sign` 和 `magnitude`。

新入口复用通用 MA 训练流程。ratio30 原双方法套件校验严格要求完整方案 B，消融组使用本说明中的专用入口。

## 汇总结果

可在训练完成后重新生成结果表，也可在两组之间查看已完成的部分。

```bash
PYTHONPATH=src python -m fedproxy.ma.ablations summarize
```

自定义目录时补充 `--reference` 和 `--root`。输出文件为 `summary.json` 和 `results.csv`，主表包含完整 MA 与四个消融组，尚未完成的组列在 `pending`，已有结果核验失败的原因保存在 `invalid`。每完成一组自动刷新一次表。评测绑定过期的组在重新启动后刷新评测，训练 checkpoint 继续复用。

结果使用第 10 轮融合后的完整 LLM。CSV 包含 ALL、QA、GLUE、八任务 accuracy、各任务下降百分点及累计上下行字节。组件贡献等于完整 MA 分数减去消融组分数，正值表示移除后退化，负值原样保留。按贡献值降序给出条件性顺序。A1 与 A3 各识别一组机制的联合贡献，单种子结果不提供跨种子置信区间或显著性结论。

上行统计实际序列化包长，下行明确记录 `downlink_accounting`。若参考 MA 是旧 full-state 传输，沿用同一配置及估算口径，表中保留该标识。初始化下载、失败轮重传和网络框架开销沿用已有账本边界。历史运行未保存的梯度、候选能量等诊断值标为不可得。

## 本地验证

```bash
PYTHONPATH=src pytest -q tests/test_ma_ablations.py tests/test_ma_protocol.py tests/test_ma_tiny_integration.py tests/test_ma_transport_resume.py tests/test_ma_lossless_transport.py tests/test_ma_config.py tests/test_ma_report.py
```

覆盖符号相反时的平均分母、交集有效性、稠密负对照、残差首轮一致性、零和非零冲突下的 PCR、随机 k 的客户端隔离、真实 MA 初始化检查入口、四组顺序执行、中断恢复及 A3 汇总。
