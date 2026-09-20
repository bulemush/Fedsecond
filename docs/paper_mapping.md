# 论文到实现的映射

| 论文位置 | 含义 | 实现 | 主要测试 |
|---|---|---|---|
| §4.1, Eq. (4) | 有效 token 上的 Block Influence | `compression/bi.py` | smoke 的 BI 路径 |
| §4.1 | 按 BI 保留 block 并保持原始层序 | `compression/prune.py` | `test_compression_fusion.py` |
| Eq. (5)-(7) | cosine、异质性、权重、冲突 | `federated/analysis.py` | `test_analysis.py` |
| Eq. (8)-(9) | PCR | `federated/regularization.py` | `test_regularization.py` |
| Eq. (10) | 异质性感知保留率 | `federated/analysis.py` | `test_analysis.py` |
| Eq. (11)-(14) | H-TIES 缩放、主导符号和更新 | `federated/aggregation.py` | `test_aggregation.py` |
| Algorithm 1 | 同 round base、本轮 h/w、下一轮 C | `federated/server.py` | server/smoke 路径 |
| §4.3, Eq. (15) | 合并 LoRA 后按映射回填原 LLM | `fusion/plug_in.py` | `test_compression_fusion.py`、smoke |
| Appendix B | AdamW、LoRA 与训练超参数 | `configs/base.yaml` | 配置校验 |

注意：论文分析发生在完整代理参数符号中，而实现根据附录的 LoRA 训练/通信描述选择 LoRA A/B 空间。这是可审计的推断实现，并非作者公开源码确认。

