# Ratio30 数据异质性实验配置设计

配置与异质性专用全流程入口已实现，启动、恢复和结果读取见 [异质性训练说明](ratio30_heterogeneous_training.md)。公共训练预算沿用刚确认的 ratio30 同质性设置：每任务最多 3000 条、输入 320 token、本地 5 epoch、10 个全局聚合轮、micro batch 4 × 累积 4。旧异质性父配置中的 5000 条和 10 本地 epoch 在本方案中明确覆盖，不会隐式沿用。

## 实验单位与数据异质性

采用现有 FedProxy 的任务级非 IID：8 个客户端各持有一个任务，联合训练一个全局模型。FedProxy 和 MA 各训练、融合一个最终模型，每个最终模型都评估八项任务，共两次独立联邦训练。

| 客户端 | 本地数据任务 | 样本上限 |
|---|---|---|
| client_0 | OBQA | 3000 |
| client_1 | ARC-Easy | 3000 |
| client_2 | ARC-Challenge | 3000 |
| client_3 | CommonsenseQA | 3000 |
| client_4 | SST-2 | 3000 |
| client_5 | MRPC | 3000 |
| client_6 | RTE | 3000 |
| client_7 | MNLI | 3000 |

各任务不足 3000 条时使用全部训练样本。全系统理论上限 24000 条，实际因部分任务规模较小而低于此值。每个客户端最多持有 3000 条，不再把每个任务的 3000 条 IID 分给四个客户端。

这里的异质性是任务、输入及标签空间存在差异，样本量也可能不同；没有引入 Dirichlet alpha，也不是在每个任务内部做标签比例偏斜。两种方法复用同一份异质性 `data_manifest.json`，固定任务顺序、所选索引及 seed 42。

本设计沿用原 FedProxy 的八任务异质性协议。同质性实验每项为四客户端，当前异质性实验为八客户端，客户端持有的样本量及模型数量也不同。因此两个场景的分数差异不能只归因于数据异质性；FedProxy 与 MA 的方法比较应分别在同一场景、同一预算内进行。如果后续要做仅改变分布的消融，需要额外统一客户端数、样本预算及模型训练方式。

## 公共配置

| 项目 | FedProxy 与 MA |
|---|---|
| 原始模型 | 服务器当前 `data/models` 中的 LLaMA-2-7B，32 层 |
| ratio | `compression.remove_ratio: 0.30` |
| 初始代理 | 删除 `floor(32 × 0.30)=9` 层，保留 23 层 |
| 实际比例 | 删除层比例 28.125%，参数删除比例另从压缩清单读取 |
| 数据 | 8 任务、8 客户端、每任务最多 3000 条、任务级非 IID |
| 输入／目标长度 | 320／128 token |
| 全局／本地预算 | 10 个聚合轮，每轮每客户端本地 5 epoch |
| Batch | micro batch 4 × 累积 4，每客户端有效 batch 16 |
| 参与 | 每轮八客户端全部参与 |
| GPU | 双卡，每卡一个客户端，一轮四批 |
| LoRA | rank 32、alpha 64、dropout 0.1，七类目标模块保持原配置 |
| 优化器 | AdamW、学习率 5e-5、weight decay 0、恒定学习率 |
| 优化器状态 | 每客户端每轮重新创建 |
| PCR | 开启，lambda 1e-5，sum reduction |
| 校准 | Alpaca 512 条，校准长度 512，batch 1，seed 42 |
| 种子 | 训练、样本选择和初始全局 LoRA 均为 42 |
| 评测 | 第 10 轮融合回 32 层原始模型，再统一评估八项任务 |
| 协议 | zero-shot、batch 1、accuracy；模型 dtype 保持现有 auto 解析规则 |
| 环境 | 沿用服务器当前 conda 环境，不新增依赖 |

320 token 与 5 本地 epoch 沿用当前预算，仍标为 diagnostic。每客户端有效 batch 为 16，不把八个客户端的 batch 相加解释为同步数据并行 batch。

## 方法差异与 MA 方案 B

FedProxy 使用原 H-TIES 与 PCR，训练 23 层代理的全部 LoRA 参数，代理骨干冻结。MA 同样加载 23 层代理，按既定方案 B 激活和上传部分层的 LoRA：

```yaml
ma:
  schedule:
    train_blocks: 17
    upload_blocks: 14
    anchor_blocks: 1
    q_block: 2
    coverage_window: 1
```

MA 其余参数与当前 `configs/ma/base.yaml` 一致：masked_sum PCR；A/B 保留率 0.8；固定稀疏率；探测率 0；bitmap 编码；active_conflict 下行；零／常量状态省略；残差 TTL 4、衰减 0.1；原有 MA 聚合超参数，包括 `n_ref=3`，不因客户端数变化另行调参。

已使用现有调度函数检查 23 层、8 客户端、seed 42 和 10 轮：每轮非锚点层实际有 4～5 个上传客户端，满足 `q_block=2`；观察到的待上传层下一次上传机会最大等待为 1 轮，满足 TTL 4。锚点层由全部八客户端上传。该结论针对层级上传机会，不代表每个稀疏坐标都有同样数量的贡献者。

末轮仍可能有未上传的残差坐标，按现有算法保存，不另增加一次强制上传或聚合，以保持方法定义不变。

## 已保存的配置

- `configs/experiments/heterogeneous_ratio30.yaml`：FedProxy。
- `configs/ma/heterogeneous_ratio30.yaml`：MA 方案 B。
- `configs/experiments/heterogeneous_ratio30_shared.yaml`：仅用于生成共享初始代理，不训练。

FedProxy 的压缩来源预留为 `runs/heterogeneous_ratio30/shared`；MA 的初始代理与数据划分来源预留为 `runs/heterogeneous_ratio30/FedProxy`。这些是初始资源复用关系，不要求复用来源完成训练或评测，更不加载其最终 adapter。

规划输出结构：

```text
runs/heterogeneous_ratio30/
  shared/
    compression/
      proxy/
      compression_manifest.json
  FedProxy/
    execution_config.yaml
    data_manifest.json
    compression -> shared 的初始代理
    checkpoints/round_0001/ ... round_0010/
    exports/llm_fused/
    evaluation/fused_summary.json
    result.json
  MA/
    execution_config.yaml
    data_manifest.json -> 同一份八客户端异质性划分
    compression -> 同一个初始代理
    clients/
    checkpoints/round_0001/ ... round_0010/
    exports/llm_fused/
    evaluation/fused_summary.json
    result.json
  experiment_manifest.json
  logs/
```

每个方法只保存一个全局融合模型，和同质性每方法八个独立模型的输出结构不同。默认使用新的总目录，已有 ratio50 实验及同质性实验保持原样。

压缩默认独立准备一份 23 层初始代理。若后续希望省略重复压缩，可复用已有 ratio30 同质性实验的未训练代理，但必须核对原始模型、校准参数、23 层结构、保留层映射及压缩清单。其 IID 数据划分和已训练 checkpoint 不用于本实验。

## 已落实的实现范围

1. 增加异质性专用的双方法入口，分别提供配置检查、共享资源准备、全流程 nohup、恢复和结果读取。不能直接调用当前 `fedproxy.ratio30` 同质性入口，其校验明确要求单任务和四客户端。
2. 对八客户端任务级划分校验任务、索引、样本上限和哈希；客户端记录为 `{task, indices}`，不能套用 IID 列表格式。
3. 两个方法绑定同一个初始代理和划分，冻结配置；校验共享压缩清单、模型和 checkpoint 的来源一致性。
4. 双卡实际可见性检查，每卡一个客户端，八客户端四批；模型初始化沿用相同 seed 42。
5. FedProxy 与 MA 可分别运行。仅运行 MA 时只准备必要输入，不强制先训练 FedProxy。
6. 从最后完整轮次恢复；MA 保留八客户端残差和暂存包。训练完成后融合最终 checkpoint、评估八项任务，各自保存结果，不增加两个方法的成绩比较拦截。
7. 每个方法输出八项 accuracy，以及 QA 四项等权平均、GLUE 四项等权平均、ALL 八项等权平均；这些分数来自同一个全局融合模型。保存原始 lm-eval JSON、评测协议、最终 checkpoint 绑定及通信统计。

实现复用现有底层训练、聚合与八任务评测代码，不修改其算法。新增入口是 `fedproxy.ratio30_heterogeneous`、`scripts/launch_ratio30_heterogeneous_suite.sh` 和 `scripts/summarize_ratio30_heterogeneous_suite.py`。真实双卡 7B 训练仍需在服务器执行，本地检查不代表训练已完成。
