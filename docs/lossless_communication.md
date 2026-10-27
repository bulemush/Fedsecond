# MA 无损通信优化

`configs/ma/base.yaml` 默认启用以下配置，继承它的变体也会继承，除非自行覆盖：

```yaml
ma:
  compression:
    index_codec: bitmap
  communication:
    downlink: active_conflict
    elide_zero_states: true
    elide_constant_states: true
```

## 三项改动

1. **上行位图**：非完整观测张量使用每个候选坐标一位的 little-bit-order 位图，数值仍为 FP32；锚点/完整观测张量保留无需索引的 dense 编码。Top-k、残差消费及上传观测集合不变。已观测的零值仍具有观测身份，不能将其改成缺失。
2. **激活冲突图下行**：服务器实际封包完整全局 P 与本轮训练层的 G，worker 解码后再开始训练。不再把完整 G 或旁路完整状态放入优化后的任务载荷。非训练层 P 仍同步，因为它们参与前向传播。封包按当前并发波次生成，避免一次保留所有客户端的下行包。
3. **精确零/常量省略**：上行每个张量的已选数值、下行 P/G 张量均可使用 zero、constant、fp32 三种格式。zero 仅适用于全为正零的原始位模式；constant 保存一次相同的 32 位模式，包含负零。混合正负零、小但非零的值不会被置零。首轮 G 为零，自动省略其数值载荷，张量描述仍在包头中。

压缩过程不改变 LoRA rank、训练层计划、样本、PCR 系数、Top-k 保留率或聚合逻辑。聚合侧始终恢复原始坐标和值。无需增加训练依赖。

## 启动独立的新实验

已经完成过 base 实验时，使用新增的独立目录配置，避免复用原来的检查点目录：

```bash
bash scripts/launch_experiment.sh configs/ma/lossless.yaml 0,1 reuse-all
```

沿用仓库已有基线压缩产物与数据复用检查。`lossless.yaml` 与 base 使用相同算法，只改变运行名称/输出目录。命令中的 GPU 编号应对应实际服务器。

需要隔离每项收益时，可以在新实验配置中覆盖：`index_codec: varint` 关闭位图；`downlink: full_state` 恢复旧完整状态传输；将两个 elide 开关设为 false 关闭数值省略。`full_state` 走旧张量下行，数值省略开关此时只影响上行。每组实验使用独立输出目录。

## 统计口径

每轮 checkpoint metadata 和 `metrics/aggregation.json` 中：

| 字段 | 含义 |
|---|---|
| `communication_bytes.uplink` | 所有客户端实际上行序列化包长度之和 |
| `communication_bytes.downlink` | 优化协议实际下行序列化包长度之和，包括 P、激活 G、包头及校验信息 |
| `communication_bytes.downlink_fp32_state_estimate` | 完整 FP32 P/G 的旧口径参考估算，新协议下不是实际下行量 |
| `communication_bytes.downlink_active_fp32_state_estimate` | 完整 P＋激活 G 未做零/常量省略、且不含包头的理论主体大小 |
| `downlink_packet_bytes` | 每客户端下行包长度 |
| `downlink_transport` | 实际传输路径标识 |

`ma-report` 优先使用实际下行字段；旧实验缺少它时回退旧估算。输出 `downlink_bytes`、`total_communication_bytes`、`downlink_accounting` 和 `downlink_full_fp32_reference_bytes`。为兼容已有消费者，`downlink_state_bytes_estimate`、`total_state_bytes_estimate` 保留为所选口径的别名；使用时必须同时查看 `downlink_accounting`。

这里只计已提交轮次的逻辑序列化载荷；未计模型首次下载、失败轮次重试、进程间 pickle 封装、TCP/TLS 等网络开销。当前仍是本机多进程仿真，不能将包长直接称为实际网卡流量。

base 的理论上行主体约为 6.315384 GiB（8 客户端×10 轮，未考虑额外零/常量省略与头部）；真正结果以包长为准。不能把先前的理论表直接当作本次实验实测值。

## 检查点和协议兼容

- 新上行 bitmap 或值省略使用 wire version 2；解码器同时接受原 version 1 的 int32/varint/Golomb 包。旧格式不能携带新编码。
- `ma_protocol_version=1` 继续表示原 MA 聚合/状态语义，和上行 wire version 是两个不同概念。下行包有独立的 version/schema、客户端、轮次、计划、base/run hash 和校验。
- 不含 `ma.communication` 的旧配置默认走旧 full_state 下行，省略开关默认关闭。继续旧实验时使用其原始 `resolved_config.train.yaml`，不能把修改后的 base 配置直接套到旧检查点；配置哈希仍会拒绝不一致恢复。
- 已暂存 update.bin 在恢复时按原字节重放，不重新训练、不再次消费残差；现有轮次 checkpoint ACK 边界不变。
- 不迁移或覆盖已有实验产物。需要比较新通信成本时使用独立的新运行目录。

## 验证

```bash
PYTHONPATH=src pytest -q tests/test_ma_lossless_transport.py tests/test_ma_transport_resume.py tests/test_ma_protocol.py tests/test_ma_tiny_integration.py tests/test_ma_config.py tests/test_ma_report.py
```

验证覆盖位图边界与填充、计数和长度损坏、正负零/小非零值、常量省略、观测零保留、下行计划及哈希、真实 server/worker 封包路径与成本统计、中断后的部分轮次重放，以及两轮 tiny Llama 的新旧路径对照。对照比较最终 LoRA、G、残差和合并模型 logits 的原始字节。

这些检查验证协议与小模型路径；完整 7B 训练准确率、GPU 性能和真实网络吞吐仍需服务器实验。
