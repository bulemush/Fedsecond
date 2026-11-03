# Ratio30 异质性全流程训练

本入口沿用服务器原 conda 环境，不新增依赖。FedProxy 和 MA 各训练一个全局模型：8 个客户端分别持有 OBQA、ARC-Easy、ARC-Challenge、CommonsenseQA、SST-2、MRPC、RTE、MNLI，所有客户端每轮共同聚合。它是任务级非 IID，不是八个独立单任务模型。

共同预算为每任务最多 3000 条、输入／目标长度 320／128 token、每轮本地 5 epoch、10 个全局轮、micro batch 4 × 累积 4。使用相同的 23 层初始代理、相同样本索引和八客户端任务划分、seed 42，以及原优化器、LoRA 和 PCR 设置。MA 使用方案 B：训练／上传／锚点 17/14/1，`q_block=2`，其余参数保持当前 MA base。

## 检查与全流程 nohup

先同步本次代码和配置到服务器，在项目根目录激活原 conda 环境，然后执行：

```bash
PYTHONPATH=src python -m fedproxy.ratio30_heterogeneous validate --gpu-ids 2,3
```

首次检查读取模型配置和权重文件状态，核对预算及八客户端调度，不生成训练输出、不加载大模型。已有实验还会检查冻结配置、共享代理、数据划分、checkpoint 和 MA 残差。成功输出应包含 `validated ... clients=8 tasks=8 rounds=10 epochs=5 MA=17/14/1`。真正启动时还会要求 PyTorch 实际识别到两张可见 GPU。

两个方法的完整后台流程：

```bash
bash scripts/launch_ratio30_heterogeneous_suite.sh 2,3 both
```

一次后台作业依次完成：

1. 生成共享初始代理，验证删除比例 0.30、原始 32 层、保留 23 层及层映射。共享代理生成后重复启动会复用。
2. 准备一次八客户端异质性划分，每客户端一个任务、最多 3000 条。两种方法引用同一份划分和同一个初始代理。
3. FedProxy：截断审计 → dry-run → 训练／续训 → 第 10 轮融合 → 同一融合模型评估八项任务 → 保存自身结果。
4. MA：按同一流程完成，沿用 17/14/1 调度和现有残差、编码及聚合实现。

全流程没有两个方法的成绩比较阶段。两张 GPU 每卡运行一个客户端，一轮八客户端分四批。压缩、审计、融合和评估按现有单进程流程运行，不要求这些阶段同时占满两张卡。

启动器会打印后台 PID 和日志路径，并显示日志。Ctrl+C 只停止日志查看，nohup 作业继续；需要时用打印的路径执行 `tail -f`。

## 单独运行与自定义目录

```bash
# 单独完成 FedProxy
bash scripts/launch_ratio30_heterogeneous_suite.sh 2,3 fedproxy

# 单独完成 MA
bash scripts/launch_ratio30_heterogeneous_suite.sh 2,3 ma

# 两个方法使用另一个新的总目录
bash scripts/launch_ratio30_heterogeneous_suite.sh 2,3 both runs/heterogeneous_ratio30_repeat
```

只运行 MA 时，会在 `FedProxy/` 中准备输入资源，但不会训练或评估 FedProxy，不要求它已有最终模型。再次启动另一个方法时复用同一份准备结果。

GPU 编号按空闲设备替换，必须是两张不同的卡。异质性入口没有单任务选项，因为当前协议要求八任务八客户端联合训练。

## 目录和结果

```text
runs/heterogeneous_ratio30/
  shared/
    execution_config.yaml
    resolved_config.compress.yaml
    compression/
      compression_manifest.json
      proxy/
  FedProxy/
    execution_config.yaml
    data_manifest.json
    compression -> 共享代理
    checkpoints/round_0001/ ... round_0010/
    exports/llm_fused/
    evaluation/
      fused_summary.json
      raw/fused_lm_eval.json
    result.json
    summary.json
    results.csv
  MA/
    execution_config.yaml
    data_manifest.json -> 同一份八客户端划分
    compression -> 同一份共享代理
    clients/
    checkpoints/
    exports/
    evaluation/
    result.json
    summary.json
    results.csv
  experiment_manifest.json
  logs/
```

每个方法完成后各自生成 `summary.json` 与 `results.csv`。CSV 包含八项任务、QA 平均、GLUE 平均和 ALL 平均，共 11 行分数；JSON 另外记录模型、预算、通信字节和耗时。这里八项分数来自同一个全局模型。通信与耗时属于整次联邦训练，不按任务重复累计。

可随时重新读取已经完成的方法：

```bash
python scripts/summarize_ratio30_heterogeneous_suite.py --method fedproxy
python scripts/summarize_ratio30_heterogeneous_suite.py --method ma
python scripts/summarize_ratio30_heterogeneous_suite.py

cat runs/heterogeneous_ratio30/FedProxy/results.csv
cat runs/heterogeneous_ratio30/MA/results.csv
```

自定义总目录时在汇总命令后添加 `--root <总目录>`。仅完成一个方法时使用对应 `--method`；默认读取两个方法。检查针对每个方法自己的最终 checkpoint、融合清单、评测绑定和八任务协议，不要求两个方法的原始评测 JSON 中模型目录字符串相同。

## 中断恢复

重新执行原启动命令即可。已完成且评测绑定正确的方法直接跳过；部分训练从最后一个完整 checkpoint 继续。MA 会恢复全部八客户端残差，第一轮尚未提交时重放已有合法暂存包；FedProxy 的未提交轮内训练重新执行。训练完成但融合或评估未完成时只补相应步骤。缺少某项分数、平均值异常或评测过期时重新评测，不重训已完成轮次。

同一总目录使用统一进程锁。保留共享代理、源模型、划分、冻结配置、`experiment_manifest.json`、checkpoint，以及 MA 的八客户端残差。输入、预算、模型或 checkpoint 不匹配会停止，避免混入其他实验产物；不通过修改历史快照跳过校验。两个不同任务中出现相同样本索引是合法的，单个任务内部的重复或超额索引会被拒绝。

本次固定 10 轮，不读取旧 ratio50 或同质性实验的已训练 adapter，不自动延长到 20 轮。旧实验目录保持原样。两场景客户端数量和训练方式不同，跨场景分数差异不能仅归因于 non-IID。

## 配置与验证

- `configs/experiments/heterogeneous_ratio30.yaml`：FedProxy。
- `configs/ma/heterogeneous_ratio30.yaml`：MA。
- `configs/experiments/heterogeneous_ratio30_shared.yaml`：仅压缩共享初始代理，不训练。

本地可运行配置、调度和替代训练器的全流程检查：

```bash
PYTHONPATH=src pytest -q tests/test_ratio30_heterogeneous.py tests/test_ratio30.py tests/test_ma_homogeneous.py tests/test_reuse.py
```

服务器原 PyTorch 环境可补充真实张量协议与 tiny 训练验证：

```bash
PYTHONPATH=src pytest -q tests/test_ma_protocol.py tests/test_ma_transport_resume.py tests/test_ma_tiny_integration.py
```

本地流程测试不代表真实 7B 训练已经完成。最终效果、显存和耗时需要在服务器记录；23 层代理的资源开销不能直接按 LoRA 激活层比例估算。
